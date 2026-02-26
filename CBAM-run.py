import os
import time
import torch
import numpy as np
import rasterio
import torch.nn.functional as F
from model_cbam_gan import UNet5DownWithGating, UnetGenerator
import functools
import torch.nn as nn
import logging
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常量定义 - 必须与训练时完全一致
ORIG_H, ORIG_W = 1207, 1563  # HR目标尺寸
PAD = 2048  # 填充尺寸
NUM_FRAMES = 3  # 3波段


class ModelInference:
    def __init__(self, model_path: str, device: str = "cuda",
                 vmin: float = None, vmax: float = None):
        """
        初始化推理类

        Args:
            model_path: 训练好的模型权重路径
            device: 推理设备 ("cuda" 或 "cpu")
            vmin, vmax: 归一化参数（如果为None则自动计算）
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vmin = vmin
        self.vmax = vmax
        self.model = self._load_model(model_path)
        logger.info(f"模型已加载到设备: {self.device}")

    def _detect_model_architecture(self, state_dict_keys: List[str]) -> str:
        """检测模型架构类型"""
        new_arch_features = [
            'ca.fc.0.weight',
            'sa.conv.weight',
            'gating.conv.0.weight',
            'attention.W_gate.0.weight',
            'bottleneck.0.conv1.weight',
            'bottleneck.1.fc.0.weight',
            'deep_supervision.0.weight'
        ]

        old_arch_features = [
            'bottleneck.conv1.weight'
        ]

        has_new_features = any(
            any(feature in key for key in state_dict_keys)
            for feature in new_arch_features
        )

        has_old_features = any(
            any(feature in key for key in state_dict_keys)
            for feature in old_arch_features
        )

        if has_new_features:
            return 'new_unet'
        elif has_old_features:
            return 'old_unet'
        else:
            logger.warning("无法确定模型架构类型，默认使用旧架构")
            return 'old_unet'

    def _create_legacy_model(self) -> nn.Module:
        """创建与旧checkpoint兼容的模型"""

        class LegacyResBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.InstanceNorm2d):
                super().__init__()
                self.same = in_ch == out_ch
                self.norm1 = norm_layer(in_ch)
                self.norm2 = norm_layer(out_ch)
                self.relu = nn.ReLU(True)
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
                self.match = nn.Identity() if self.same else nn.Conv2d(in_ch, out_ch, 1, bias=True)

            def forward(self, x):
                residual = self.match(x)
                out = self.relu(self.norm1(x))
                out = self.conv1(out)
                out = self.relu(self.norm2(out))
                out = self.conv2(out)
                out = self.relu(out + residual)
                return out

        class LegacyDownBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.res = LegacyResBlock(in_ch, out_ch)
                self.pool = nn.MaxPool2d(2, 2)

            def forward(self, x):
                feat = self.res(x)
                down = self.pool(feat)
                return down, feat

        class LegacyUpBlock(nn.Module):
            def __init__(self, prev_ch: int, skip_ch: int, out_ch: int):
                super().__init__()
                self.up = nn.ConvTranspose2d(prev_ch, skip_ch, 2, stride=2)
                self.res = LegacyResBlock(skip_ch * 2, out_ch)

            def forward(self, x, skip):
                x = self.up(x)
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bicubic", align_corners=False)
                x = torch.cat([skip, x], dim=1)
                return self.res(x)

        class LegacyUNet(nn.Module):
            def __init__(self, input_nc: int = 3, output_nc: int = 3):
                super().__init__()
                ch = [32, 64, 128, 256, 512]
                self.d1 = LegacyDownBlock(input_nc, ch[0])
                self.d2 = LegacyDownBlock(ch[0], ch[1])
                self.d3 = LegacyDownBlock(ch[1], ch[2])
                self.d4 = LegacyDownBlock(ch[2], ch[3])
                self.bottleneck = LegacyResBlock(ch[3], ch[4])
                self.u1 = LegacyUpBlock(ch[4], ch[3], ch[3])
                self.u2 = LegacyUpBlock(ch[3], ch[2], ch[2])
                self.u3 = LegacyUpBlock(ch[2], ch[1], ch[1])
                self.u4 = LegacyUpBlock(ch[1], ch[0], ch[0])
                self.out_conv = nn.Sequential(
                    nn.Conv2d(ch[0], output_nc, 1),
                    nn.Tanh()
                )

            def forward(self, x):
                d1, s1 = self.d1(x)
                d2, s2 = self.d2(d1)
                d3, s3 = self.d3(d2)
                d4, s4 = self.d4(d3)
                bott = self.bottleneck(d4)
                u1 = self.u1(bott, s4)
                u2 = self.u2(u1, s3)
                u3 = self.u3(u2, s2)
                u4 = self.u4(u3, s1)
                return self.out_conv(u4)

        return LegacyUNet(
            input_nc=NUM_FRAMES,
            output_nc=NUM_FRAMES
        ).to(self.device)

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'generator_state_dict' in checkpoint:
            generator_state_dict = checkpoint['generator_state_dict']
        elif 'model_state_dict' in checkpoint:
            generator_state_dict = checkpoint['model_state_dict']
        else:
            generator_state_dict = checkpoint

        state_dict_keys = list(generator_state_dict.keys())
        arch_type = self._detect_model_architecture(state_dict_keys)

        logger.info(f"检测到模型架构类型: {arch_type}")

        if arch_type == 'new_unet':
            model = UNet5DownWithGating(
                input_nc=NUM_FRAMES,
                output_nc=NUM_FRAMES,
                attention_type="simple",
                use_multi_scale=False
            ).to(self.device)
            logger.info("使用新的增强UNet架构")
        else:
            model = self._create_legacy_model()
            logger.info("使用旧的兼容UNet架构")

        try:
            model.load_state_dict(generator_state_dict)
            logger.info("成功加载模型权重")
        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in generator_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"部分加载模型权重，加载了 {len(pretrained_dict)} 个参数")

        model.eval()
        return model

    def _read_tif(self, path: str) -> Tuple[np.ndarray, dict]:
        """读取TIF文件并返回数据和元信息（与训练代码一致）"""
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

            # 处理NoData值
            if nodata is not None:
                logger.debug(f"检测到NoData值: {nodata}，替换为NaN")
                data[data == nodata] = np.nan

            # 处理常见NoData标记
            if np.any(data < -999):
                logger.debug(f"检测到疑似NoData值 (< -999)，替换为NaN")
                data[data < -999] = np.nan

        return data, {'meta': meta, 'transform': transform, 'crs': crs}

    def _write_tif(self, data: np.ndarray, output_path: str, reference_info: dict):
        """保存数据为TIF文件"""
        meta = reference_info['meta'].copy()
        meta.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'dtype': data.dtype
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data, 1)
            dst.transform = reference_info['transform']
            dst.crs = reference_info['crs']

    def _compute_norm_params(self, images: List[np.ndarray]) -> Tuple[float, float]:
        """计算归一化参数（仅基于有效数据）"""
        all_valid_data = []
        for img in images:
            valid_data = img[np.isfinite(img)]
            if len(valid_data) > 0:
                all_valid_data.append(valid_data)

        if not all_valid_data:
            raise ValueError("所有输入图像都没有有效数据")

        all_valid = np.concatenate(all_valid_data)
        vmin, vmax = float(all_valid.min()), float(all_valid.max())

        return vmin, vmax

    def _preprocess_batch(self, tif_paths: List[str], reference_info: dict) -> Tuple[torch.Tensor, float, float]:
        """
        预处理一个批次的TIF文件（完全按照训练代码的流程）

        Returns:
            processed_tensor: 预处理后的张量 (1, 3, PAD, PAD)
            vmin, vmax: 归一化参数
        """
        assert len(tif_paths) == NUM_FRAMES, f"需要 {NUM_FRAMES} 张图像，但提供了 {len(tif_paths)} 张"

        # 1. 读取所有LR图像
        lr_images = []
        for path in tif_paths:
            data, _ = self._read_tif(path)
            lr_images.append(data)
            logger.info(f"读取LR图像: {os.path.basename(path)}, 尺寸: {data.shape}")

        # 2. 计算归一化参数（如果未指定）
        if self.vmin is None or self.vmax is None:
            vmin, vmax = self._compute_norm_params(lr_images)
            logger.info(f"自动计算归一化参数: vmin={vmin:.6f}, vmax={vmax:.6f}")
        else:
            vmin, vmax = self.vmin, self.vmax
            logger.info(f"使用指定归一化参数: vmin={vmin:.6f}, vmax={vmax:.6f}")

        processed_images = []

        for i, lr in enumerate(lr_images):
            # 3. 处理NaN值（用有效数据的中位数填充）
            if np.any(np.isnan(lr)):
                valid_mask = np.isfinite(lr)
                if np.any(valid_mask):
                    fill_value = np.median(lr[valid_mask])
                    logger.info(f"图像{i + 1}: 用中位数 {fill_value:.6f} 填充 {np.isnan(lr).sum()} 个NaN值")
                    lr = np.where(np.isnan(lr), fill_value, lr)
                else:
                    raise ValueError(f"图像{i + 1}全部是NaN")

            # 4. 确保非负（与训练代码一致）
            lr = np.maximum(lr, 0)

            # 5. 上采样到HR尺寸（与训练代码完全一致）
            lr_tensor = torch.from_numpy(lr)[None, None].float()
            lr_up = F.interpolate(
                lr_tensor,
                size=(ORIG_H, ORIG_W),
                mode="bicubic",
                align_corners=False
            ).squeeze().numpy()

            logger.info(f"图像{i + 1}: 上采样 {lr.shape} -> {lr_up.shape}")

            # 6. 归一化到 [-1, 1]（与训练代码完全一致）
            lr_norm = (lr_up - vmin) / (vmax - vmin + 1e-8) * 2 - 1

            # 7. 填充到PAD尺寸（与训练代码完全一致）
            padded = np.zeros((PAD, PAD), dtype=np.float32)
            padded[:ORIG_H, :ORIG_W] = lr_norm

            processed_images.append(padded)

            logger.info(f"图像{i + 1}: 归一化后范围 [{lr_norm.min():.4f}, {lr_norm.max():.4f}]")

        # 8. 堆叠成批次
        batch_tensor = torch.from_numpy(np.stack(processed_images, axis=0))[None]  # (1, 3, PAD, PAD)

        logger.info(f"批次张量形状: {batch_tensor.shape}")

        return batch_tensor, vmin, vmax

    def _postprocess_batch(self, output_tensor: torch.Tensor, vmin: float, vmax: float) -> List[np.ndarray]:
        """
        后处理模型输出（与训练代码的反向过程一致）

        Args:
            output_tensor: 模型输出 (1, 3, PAD, PAD)
            vmin, vmax: 反归一化参数

        Returns:
            List[np.ndarray]: 后处理后的图像列表
        """
        output_np = output_tensor.squeeze(0).detach().cpu().numpy()  # (3, PAD, PAD)

        processed_images = []
        for i in range(NUM_FRAMES):
            img = output_np[i]  # (PAD, PAD)

            # 1. 裁剪到原始尺寸
            img_cropped = img[:ORIG_H, :ORIG_W]

            # 2. 反归一化: [-1, 1] -> [vmin, vmax]
            img_denorm = (img_cropped + 1) / 2 * (vmax - vmin + 1e-8) + vmin

            # 3. 转换为float32（避免float16溢出）
            img_final = img_denorm.astype(np.float32)

            logger.info(f"输出图像{i + 1}: 范围 [{img_final.min():.4f}, {img_final.max():.4f}]")

            processed_images.append(img_final)

        return processed_images

    @torch.no_grad()
    def predict_images(self, input_tif_paths: List[str], reference_tif_path: str,
                       output_dir: str) -> List[str]:
        """
        对输入的TIF序列进行超分辨率处理

        Args:
            input_tif_paths: 输入TIF文件路径列表（按时间顺序）
            reference_tif_path: 参考TIF文件路径（用于获取投影信息）
            output_dir: 输出目录

        Returns:
            List[str]: 输出文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取参考图像的元信息
        _, reference_info = self._read_tif(reference_tif_path)

        num_images = len(input_tif_paths)
        output_paths = []

        # 每三张图像处理
        for i in range(0, num_images, NUM_FRAMES):
            current_window = input_tif_paths[i:i + NUM_FRAMES]

            if len(current_window) < NUM_FRAMES:
                logger.warning(f"跳过不足 {NUM_FRAMES} 张图像的批次")
                continue

            logger.info(f"\n{'#' * 70}")
            logger.info(f"处理批次 {i // NUM_FRAMES + 1}/{(num_images + NUM_FRAMES - 1) // NUM_FRAMES}")
            logger.info(f"输入文件: {[os.path.basename(p) for p in current_window]}")
            logger.info(f"{'#' * 70}")

            try:
                # 预处理
                input_tensor, vmin, vmax = self._preprocess_batch(current_window, reference_info)
                input_tensor = input_tensor.to(self.device)

                # 模型推理
                logger.info("开始模型推理...")
                with torch.cuda.amp.autocast():
                    output_tensor = self.model(input_tensor)
                logger.info("模型推理完成")

                # 后处理
                output_images = self._postprocess_batch(output_tensor, vmin, vmax)

                # 保存输出图像
                for j, output_img in enumerate(output_images):
                    output_filename = f"sr_{os.path.basename(current_window[j])}"
                    output_path = os.path.join(output_dir, output_filename)

                    self._write_tif(output_img, output_path, reference_info)
                    output_paths.append(output_path)

                    logger.info(f"✅ 保存输出: {output_path}")

            except Exception as e:
                logger.error(f"❌ 处理批次失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        return output_paths

    def inspect_checkpoint(self, model_path: str):
        """检查checkpoint文件的内容"""
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info("\nCheckpoint内容:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                logger.info(f"  {key}: dict with {len(checkpoint[key])} keys")
            elif hasattr(checkpoint[key], 'shape'):
                logger.info(f"  {key}: tensor with shape {checkpoint[key].shape}")
            else:
                logger.info(f"  {key}: {type(checkpoint[key])}")

        if 'generator_state_dict' in checkpoint:
            logger.info(f"\ngenerator_state_dict键 (前10个):")
            gen_keys = list(checkpoint['generator_state_dict'].keys())
            for key in gen_keys[:10]:
                logger.info(f"    {key}")

        if 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        arch_type = self._detect_model_architecture(list(state_dict.keys()))
        logger.info(f"\n检测到的架构类型: {arch_type}\n")


def main():
    """使用示例"""
    # ========== 配置区 ==========
    model_path = r"D:\Bestepoch\BESTMODEL_CBAMGAN.pth"
    input_dir = r"D:\Data\Test\100m\max"
    output_dir = r"D:\Data\Test\Result"
    reference_tif_path = r"D:\Data\Train\10m\Chir04100a_0001.tif"

    # 归一化参数（可选，如果为None则自动计算）
    vmin = None  # 例如: 0.0
    vmax = None  # 例如: 1.0

    logger.info(f"\n{'=' * 70}")
    logger.info(f"配置:")
    logger.info(f"  模型路径: {model_path}")
    logger.info(f"  输入目录: {input_dir}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  参考图像: {reference_tif_path}")
    logger.info(f"  归一化参数: vmin={vmin}, vmax={vmax}")
    logger.info(f"{'=' * 70}\n")

    # 检查checkpoint
    temp_inference = ModelInference.__new__(ModelInference)
    temp_inference.inspect_checkpoint(model_path)

    # 获取输入文件
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]
    input_files.sort()

    if len(input_files) % NUM_FRAMES != 0:
        logger.warning(f"输入文件数量 {len(input_files)} 不是 {NUM_FRAMES} 的倍数")

    logger.info(f"找到 {len(input_files)} 个输入文件\n")

    # 创建推理实例
    inference = ModelInference(model_path, device="cuda", vmin=vmin, vmax=vmax)

    # 开始推理
    start_time = time.time()
    output_paths = inference.predict_images(input_files, reference_tif_path, output_dir)
    end_time = time.time()

    duration = end_time - start_time

    logger.info("\n" + "=" * 70)
    logger.info(f"✅ 推理完成！")
    logger.info(f"   共生成 {len(output_paths)} 个文件")
    logger.info(f"   总运行时间: {duration:.2f} 秒")
    logger.info(f"   平均每个文件: {duration / len(output_paths):.2f} 秒")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()