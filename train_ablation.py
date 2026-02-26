"""
优化的统一消融实验训练系统
- 移除重复的SR-GAN模型
- 优化4个模型的清晰对比
- 改进训练流程和结果展示
- 增强错误处理和日志记录
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse_metric
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from data_preprocessing import HydrologyDataProcessor, NUM_FRAMES
from model_registry import (
    MODEL_REGISTRY,
    create_generator,
    create_discriminator,
    create_loss_function,
    print_model_summary
)

# ========================= 全局配置 ========================= #
PATHS = {
    "train_hr": r"D:\Data\Train\10m",
    "train_lr": r"D:\Data\Train\10m",
    "save_dir": r"D:\Data\Train\ablation_results",
    "log_dir": r"D:\Data\Train\ablation_logs",
}

TRAIN_CONFIG = {
    "batch_size": 2,
    "epochs": 100,
    "lr_g": 1e-3,
    "lr_d": 1e-3,
    "betas": (0.5, 0.999),
    "d_steps": 1,
    "g_steps": 2,
    "val_interval": 1,
    "save_interval": 50,
    "early_stop_patience": 10,
    "gradient_clip": 1.0,
}

MODEL_CONFIG = {
    "input_nc": NUM_FRAMES,
    "output_nc": NUM_FRAMES,
    "ngf": 32,
    "ndf": 32,
}


class SingleModelTrainer:
    """单个模型的训练器 - 优化版"""

    def __init__(self, model_id: str, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 save_dir: str, log_dir: str):

        self.model_id = model_id
        self.config = MODEL_REGISTRY[model_id]
        self.device = device

        # 创建模型专用目录
        self.save_dir = os.path.join(save_dir, model_id)
        self.log_dir = os.path.join(log_dir, model_id)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志
        self._setup_logger()

        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 初始化模型
        self._init_model()

        # 训练历史
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_mse": [],
            "val_psnr": [],
            "val_ssim": [],
            "lr": [],
            "epoch_time": []
        }
        if self.config["use_gan"]:
            self.history.update({
                "train_g_loss": [],
                "train_d_loss": [],
                "d_real_score": [],
                "d_fake_score": []
            })

        self.best_mse = float('inf')
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_epoch = 0
        self.no_improve_count = 0

        self.logger.info(f"[INIT] Model: {self.config['name']}")
        self.logger.info(f"[INIT] Description: {self.config['description']}")
        self.logger.info(f"[INIT] Purpose: {self.config['purpose']}")

    def _setup_logger(self):
        """设置日志系统 - 修复Windows编码问题"""
        log_file = os.path.join(self.log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        self.logger = logging.getLogger(f"Trainer-{self.model_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # 文件处理器 - 使用UTF-8编码
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)

        # 控制台处理器 - 使用UTF-8编码并捕获错误
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 尝试设置控制台为UTF-8
        try:
            import sys
            if sys.stdout.encoding != 'utf-8':
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

        # 简化格式，移除可能导致编码问题的emoji
        formatter = logging.Formatter(
            f'%(asctime)s | [{self.model_id}] %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _init_model(self):
        """初始化模型、优化器和损失函数"""
        # 生成器
        self.G = create_generator(
            self.model_id,
            input_nc=MODEL_CONFIG["input_nc"],
            output_nc=MODEL_CONFIG["output_nc"],
            ngf=MODEL_CONFIG["ngf"]
        ).to(self.device)

        g_params = sum(p.numel() for p in self.G.parameters())
        self.logger.info(f"[Generator] Parameters: {g_params:,}")

        # 判别器 (仅GAN模型)
        if self.config["use_gan"]:
            self.D = create_discriminator(
                ndf=MODEL_CONFIG["ndf"],
                input_nc=MODEL_CONFIG["output_nc"]
            ).to(self.device)

            d_params = sum(p.numel() for p in self.D.parameters())
            self.logger.info(f"[Discriminator] Parameters: {d_params:,}")

            # 损失函数
            self.criterion = create_loss_function(self.model_id, self.D).to(self.device)

            # 优化器
            self.optG = optim.Adam(
                self.G.parameters(),
                lr=TRAIN_CONFIG["lr_g"],
                betas=TRAIN_CONFIG["betas"]
            )
            self.optD = optim.Adam(
                self.D.parameters(),
                lr=TRAIN_CONFIG["lr_d"],
                betas=TRAIN_CONFIG["betas"]
            )

            # AMP Scaler
            self.scalerG = GradScaler()
            self.scalerD = GradScaler()

        else:
            # 非GAN模型
            self.criterion = create_loss_function(self.model_id)
            self.optG = optim.Adam(
                self.G.parameters(),
                lr=TRAIN_CONFIG["lr_g"],
                betas=TRAIN_CONFIG["betas"]
            )
            self.scaler = GradScaler()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optG,
            mode='min',
            factor=0.8,
            patience=15,
            min_lr=1e-6,
            verbose=True
        )

    def _train_epoch_gan(self, epoch: int) -> Tuple[float, float, float, float]:
        """GAN模型训练 - 优化版"""
        self.G.train()
        self.D.train()

        g_loss_total, d_loss_total = 0.0, 0.0
        real_scores, fake_scores = [], []
        batch_count = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[{self.model_id}] Epoch {epoch:03d}",
            leave=False
        )

        for batch in pbar:
            if batch is None:
                continue

            try:
                lr_img = batch["input"].to(self.device, non_blocking=True)
                hr_img = batch["target"].to(self.device, non_blocking=True)
                mask = batch["valid_mask"].to(self.device, non_blocking=True)

                # ============ 更新判别器 ============
                d_loss_batch = 0.0
                for _ in range(TRAIN_CONFIG["d_steps"]):
                    self.optD.zero_grad(set_to_none=True)

                    with autocast():
                        with torch.no_grad():
                            fake = self.G(lr_img).detach()

                        real_pred = self.D(hr_img)
                        fake_pred = self.D(fake)

                        d_loss_dict = self.criterion.compute_discriminator_loss(
                            real_pred, fake_pred, hr_img, fake
                        )

                    d_loss = d_loss_dict['total']

                    if torch.isfinite(d_loss):
                        self.scalerD.scale(d_loss).backward()
                        self.scalerD.unscale_(self.optD)
                        torch.nn.utils.clip_grad_norm_(
                            self.D.parameters(),
                            max_norm=TRAIN_CONFIG["gradient_clip"]
                        )
                        self.scalerD.step(self.optD)
                        self.scalerD.update()

                        d_loss_batch += d_loss.item()
                        real_scores.append(d_loss_dict['real_score'])
                        fake_scores.append(d_loss_dict['fake_score'])

                # ============ 更新生成器 ============
                g_loss_batch = 0.0
                for _ in range(TRAIN_CONFIG["g_steps"]):
                    self.optG.zero_grad(set_to_none=True)

                    with autocast():
                        fake = self.G(lr_img)
                        fake_pred = self.D(fake)

                        g_loss_dict = self.criterion.compute_generator_loss(
                            fake_pred, hr_img, fake, mask
                        )

                    g_loss = g_loss_dict['total']

                    if torch.isfinite(g_loss):
                        self.scalerG.scale(g_loss).backward()
                        self.scalerG.unscale_(self.optG)
                        torch.nn.utils.clip_grad_norm_(
                            self.G.parameters(),
                            max_norm=TRAIN_CONFIG["gradient_clip"]
                        )
                        self.scalerG.step(self.optG)
                        self.scalerG.update()

                        g_loss_batch += g_loss.item()

                # 累计损失
                if g_loss_batch > 0 and d_loss_batch > 0:
                    g_loss_total += g_loss_batch / TRAIN_CONFIG["g_steps"]
                    d_loss_total += d_loss_batch / TRAIN_CONFIG["d_steps"]
                    batch_count += 1

                    pbar.set_postfix({
                        'G': f'{g_loss_batch / TRAIN_CONFIG["g_steps"]:.3f}',
                        'D': f'{d_loss_batch / TRAIN_CONFIG["d_steps"]:.3f}',
                        'R': f'{real_scores[-1]:.2f}' if real_scores else '0',
                        'F': f'{fake_scores[-1]:.2f}' if fake_scores else '0'
                    })

            except Exception as e:
                self.logger.warning(f"Batch training error: {e}")
                continue

        if batch_count == 0:
            return float('inf'), float('inf'), 0.0, 0.0

        return (
            g_loss_total / batch_count,
            d_loss_total / batch_count,
            np.mean(real_scores) if real_scores else 0.0,
            np.mean(fake_scores) if fake_scores else 0.0
        )

    def _train_epoch_simple(self, epoch: int) -> float:
        """非GAN模型训练 - 优化版"""
        self.G.train()

        loss_total = 0.0
        batch_count = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[{self.model_id}] Epoch {epoch:03d}",
            leave=False
        )

        for batch in pbar:
            if batch is None:
                continue

            try:
                lr_img = batch["input"].to(self.device, non_blocking=True)
                hr_img = batch["target"].to(self.device, non_blocking=True)
                mask = batch["valid_mask"].to(self.device, non_blocking=True)

                self.optG.zero_grad(set_to_none=True)

                with autocast():
                    pred = self.G(lr_img)
                    loss_dict = self.criterion.compute_generator_loss(
                        None, hr_img, pred, mask
                    )

                loss = loss_dict['total']

                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optG)
                    torch.nn.utils.clip_grad_norm_(
                        self.G.parameters(),
                        max_norm=TRAIN_CONFIG["gradient_clip"]
                    )
                    self.scaler.step(self.optG)
                    self.scaler.update()

                    loss_total += loss.item()
                    batch_count += 1

                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            except Exception as e:
                self.logger.warning(f"Batch training error: {e}")
                continue

        return loss_total / batch_count if batch_count > 0 else float('inf')

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, float, float]:
        """验证 - 优化版"""
        self.G.eval()

        psnr_list, ssim_list, mse_list = [], [], []

        for batch in tqdm(self.val_loader, desc=f"[{self.model_id}] Validating", leave=False):
            if batch is None:
                continue

            try:
                lr_img = batch["input"].to(self.device, non_blocking=True)
                hr_img = batch["target"].to(self.device, non_blocking=True)
                mask = batch["valid_mask"].to(self.device, non_blocking=True)

                with autocast():
                    pred = self.G(lr_img)

                # 转numpy
                pred_np = pred.clamp(-1, 1).cpu().numpy()
                hr_np = hr_img.cpu().numpy()
                mask_np = mask.cpu().numpy()

                # 处理维度
                if pred_np.ndim == 5:
                    B, N, C, H, W = pred_np.shape
                    pred_np = pred_np.reshape(-1, C, H, W)
                    hr_np = hr_np.reshape(-1, C, H, W)
                    mask_np = np.repeat(mask_np, N, axis=0)

                B, C, H, W = pred_np.shape

                # 计算指标
                for b in range(B):
                    for c in range(C):
                        valid_mask = mask_np[b, 0] == 1
                        if np.sum(valid_mask) < 10:  # 最少10个像素
                            continue

                        pred_valid = pred_np[b, c][valid_mask]
                        hr_valid = hr_np[b, c][valid_mask]

                        if len(pred_valid) > 0:
                            try:
                                psnr_val = psnr(hr_valid, pred_valid, data_range=2.0)
                                ssim_val = ssim(hr_valid, pred_valid, data_range=2.0)
                                mse_val = mse_metric(hr_valid, pred_valid)

                                if np.isfinite([psnr_val, ssim_val, mse_val]).all():
                                    psnr_list.append(psnr_val)
                                    ssim_list.append(ssim_val)
                                    mse_list.append(mse_val)
                            except:
                                continue

            except Exception as e:
                self.logger.warning(f"Validation batch error: {e}")
                continue

        if not psnr_list:
            return float('inf'), 0.0, 0.0

        return np.mean(mse_list), np.mean(psnr_list), np.mean(ssim_list)

    def train(self):
        """主训练循环"""
        self.logger.info("=" * 80)
        self.logger.info(f"[START] Training: {self.config['name']}")
        self.logger.info("=" * 80)

        start_time = time.time()

        for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
            epoch_start = time.time()

            # 初始化损失变量
            g_loss = 0.0
            d_loss = 0.0
            real_score = 0.0
            fake_score = 0.0
            train_loss = 0.0

            # 训练
            if self.config["use_gan"]:
                g_loss, d_loss, real_score, fake_score = self._train_epoch_gan(epoch)
                train_loss = g_loss

                self.history["train_g_loss"].append(g_loss)
                self.history["train_d_loss"].append(d_loss)
                self.history["d_real_score"].append(real_score)
                self.history["d_fake_score"].append(fake_score)
            else:
                train_loss = self._train_epoch_simple(epoch)

            self.history["train_loss"].append(train_loss)

            # 验证
            if epoch % TRAIN_CONFIG["val_interval"] == 0:
                val_mse, val_psnr, val_ssim = self._validate(epoch)

                epoch_time = time.time() - epoch_start

                self.history["epoch"].append(epoch)
                self.history["val_mse"].append(val_mse)
                self.history["val_psnr"].append(val_psnr)
                self.history["val_ssim"].append(val_ssim)
                self.history["lr"].append(self.optG.param_groups[0]['lr'])
                self.history["epoch_time"].append(epoch_time)

                # 日志
                if self.config["use_gan"]:
                    self.logger.info(
                        f"Epoch {epoch:03d}/{TRAIN_CONFIG['epochs']} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"G: {g_loss:.4f} | D: {d_loss:.4f} | "
                        f"MSE: {val_mse:.5f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch:03d}/{TRAIN_CONFIG['epochs']} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"Loss: {train_loss:.4f} | "
                        f"MSE: {val_mse:.5f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
                    )

                # 保存最佳模型
                if val_mse < self.best_mse:
                    self.best_mse = val_mse
                    self.best_psnr = val_psnr
                    self.best_ssim = val_ssim
                    self.best_epoch = epoch
                    self.no_improve_count = 0

                    self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(
                        f">>> NEW BEST MODEL! MSE: {val_mse:.5f} | "
                        f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
                    )
                else:
                    self.no_improve_count += 1

                # 学习率调整
                old_lr = self.optG.param_groups[0]['lr']
                self.scheduler.step(val_mse)
                new_lr = self.optG.param_groups[0]['lr']
                if new_lr != old_lr:
                    self.logger.info(f"[LR] Learning rate adjusted: {old_lr:.2e} -> {new_lr:.2e}")

                # 早停
                if self.no_improve_count >= TRAIN_CONFIG["early_stop_patience"]:
                    self.logger.info(f"[STOP] Early stopping at epoch {epoch}")
                    break

            # 定期保存
            if epoch % TRAIN_CONFIG["save_interval"] == 0:
                self._save_checkpoint(epoch, is_best=False)

        # 训练结束
        total_time = time.time() - start_time
        self.logger.info("=" * 80)
        self.logger.info(f"[COMPLETE] Training finished in {total_time / 3600:.2f} hours")
        self.logger.info(f"[BEST] Epoch: {self.best_epoch} | MSE: {self.best_mse:.5f} | "
                         f"PSNR: {self.best_psnr:.2f} | SSIM: {self.best_ssim:.4f}")
        self.logger.info("=" * 80)

        # 保存历史
        self._save_history()

        return self.history

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """保存检查点 - 确保唯一命名"""
        checkpoint = {
            "model_id": self.model_id,
            "model_name": self.config["name"],
            "epoch": epoch,
            "generator_state_dict": self.G.state_dict(),
            "optimizer_G_state_dict": self.optG.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_mse": self.best_mse,
            "best_psnr": self.best_psnr,
            "best_ssim": self.best_ssim,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": self.config
        }

        if self.config["use_gan"]:
            checkpoint.update({
                "discriminator_state_dict": self.D.state_dict(),
                "optimizer_D_state_dict": self.optD.state_dict()
            })

        if is_best:
            # 最佳模型：模型名_best.pth
            path = os.path.join(self.save_dir, f"{self.model_id}_best.pth")
            torch.save(checkpoint, path)
            self.logger.info(f"[SAVE] Best model saved: {os.path.basename(path)}")
        else:
            # 定期保存：模型名_epoch_XXX.pth
            path = os.path.join(self.save_dir, f"{self.model_id}_epoch_{epoch:03d}.pth")
            torch.save(checkpoint, path)
            self.logger.info(f"[SAVE] Checkpoint saved: {os.path.basename(path)}")

    def _save_history(self):
        """保存训练历史 - 使用唯一命名"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON格式：模型名_history.json
        history_json = os.path.join(self.save_dir, f"{self.model_id}_training_history.json")
        with open(history_json, 'w') as f:
            json.dump(self.history, f, indent=2)

        # CSV格式：模型名_history.csv
        if self.history["epoch"]:
            df = pd.DataFrame(self.history)
            history_csv = os.path.join(self.save_dir, f"{self.model_id}_training_history.csv")
            df.to_csv(history_csv, index=False)

            # 额外保存一份带时间戳的备份
            backup_csv = os.path.join(self.save_dir, f"{self.model_id}_history_{timestamp}.csv")
            df.to_csv(backup_csv, index=False)

        self.logger.info(f"[SAVE] Training history saved")


class AblationExperiment:
    """消融实验管理器 - 优化版"""

    def __init__(self, model_ids: Optional[List[str]] = None):
        """
        Args:
            model_ids: 要训练的模型ID列表,默认训练所有4个模型
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'=' * 80}")
        print(f"🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"{'=' * 80}\n")

        # 确定要训练的模型
        if model_ids is None:
            self.model_ids = list(MODEL_REGISTRY.keys())
        else:
            self.model_ids = [mid for mid in model_ids if mid in MODEL_REGISTRY]

        if not self.model_ids:
            raise ValueError("No valid model IDs provided")

        print(f"📋 Models to train: {self.model_ids}\n")

        # 创建目录
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)

        # 初始化数据
        self._init_data()

        # 结果汇总
        self.results = {}

    def _init_data(self):
        """初始化数据加载器"""
        print("📊 Initializing dataset...")

        full_ds = HydrologyDataProcessor(
            hr_root=PATHS["train_hr"],
            lr_root=PATHS["train_lr"],
            paths=PATHS
        )

        # 划分数据集
        val_len = int(0.2 * len(full_ds))
        train_len = len(full_ds) - val_len
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_ds,
            batch_size=TRAIN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=full_ds.collate_fn,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=full_ds.collate_fn,
            pin_memory=True
        )

        print(f"✅ Dataset size - Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    def run(self):
        """运行完整的消融实验"""
        print("\n" + "=" * 80)
        print("🔬 STARTING ABLATION STUDY (4 Models)")
        print("=" * 80)
        print_model_summary()

        total_start = time.time()
        successful_models = []
        failed_models = []

        # 训练每个模型
        for i, model_id in enumerate(self.model_ids, 1):
            print(f"\n{'=' * 80}")
            print(f"🎯 Training Model {i}/{len(self.model_ids)}: {model_id}")
            print(f"{'=' * 80}\n")

            try:
                trainer = SingleModelTrainer(
                    model_id=model_id,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    device=self.device,
                    save_dir=PATHS["save_dir"],
                    log_dir=PATHS["log_dir"]
                )

                history = trainer.train()

                # 保存结果
                self.results[model_id] = {
                    "best_mse": trainer.best_mse,
                    "best_psnr": trainer.best_psnr,
                    "best_ssim": trainer.best_ssim,
                    "best_epoch": trainer.best_epoch,
                    "history": history,
                    "status": "success"
                }

                successful_models.append(model_id)
                print(f"\n[SUCCESS] {model_id} training completed!\n")

            except Exception as e:
                print(f"\n[ERROR] Training {model_id} failed: {e}")
                import traceback
                traceback.print_exc()

                failed_models.append(model_id)
                self.results[model_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                continue

        # 实验结束
        total_time = time.time() - total_start
        print(f"\n{'=' * 80}")
        print(f"[FINISH] ABLATION STUDY COMPLETED")
        print(f"[TIME] Total time: {total_time / 3600:.2f} hours")
        print(f"[RESULT] Successful: {len(successful_models)}/{len(self.model_ids)}")
        if failed_models:
            print(f"[RESULT] Failed: {', '.join(failed_models)}")
        print(f"{'=' * 80}\n")

        # 生成对比报告
        if successful_models:
            self._generate_report()

    def _generate_report(self):
        """生成实验对比报告"""
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS COMPARISON")
        print("=" * 80)

        # 提取成功模型的结果
        comparison = []
        for model_id, result in self.results.items():
            if result.get("status") != "success":
                continue

            config = MODEL_REGISTRY[model_id]
            comparison.append({
                "Model ID": model_id,
                "Model Name": config["name"],
                "Best Epoch": result["best_epoch"],
                "Best MSE": result["best_mse"],
                "Best PSNR": result["best_psnr"],
                "Best SSIM": result["best_ssim"],
                "Attention": "✓" if config["use_attention"] else "✗",
                "GAN": "✓" if config["use_gan"] else "✗",
                "Purpose": config["purpose"]
            })

        if not comparison:
            print("No successful models to compare")
            return

        # 创建DataFrame
        df = pd.DataFrame(comparison)

        # 按MSE排序
        df = df.sort_values("Best MSE")

        # 打印表格
        print("\n" + df.to_string(index=False))
        print("\n" + "=" * 80)

        # 找出最佳模型
        best_model = df.iloc[0]
        print(f"\n[BEST MODEL] {best_model['Model Name']}")
        print(f"  MSE: {best_model['Best MSE']:.5f}")
        print(f"  PSNR: {best_model['Best PSNR']:.2f}")
        print(f"  SSIM: {best_model['Best SSIM']:.4f}")

        # 分析改进
        print(f"\n[ABLATION ANALYSIS]")
        print("-" * 80)

        # 尝试找到各个模型进行对比
        models = {row['Model ID']: row for _, row in df.iterrows()}

        if 'pure_unet' in models and 'cbam_unet' in models:
            mse_improve = (models['pure_unet']['Best MSE'] - models['cbam_unet']['Best MSE']) / models['pure_unet']['Best MSE'] * 100
            print(f"  CBAM Attention Effect (pure_unet → cbam_unet):")
            print(f"    MSE improvement: {mse_improve:+.2f}%")

        if 'pure_unet' in models and 'unet_gan' in models:
            mse_improve = (models['pure_unet']['Best MSE'] - models['unet_gan']['Best MSE']) / models['pure_unet']['Best MSE'] * 100
            print(f"  GAN + Complex Loss Effect (pure_unet → unet_gan):")
            print(f"    MSE improvement: {mse_improve:+.2f}%")

        if 'cbam_unet' in models and 'cbam_gan' in models:
            mse_improve = (models['cbam_unet']['Best MSE'] - models['cbam_gan']['Best MSE']) / models['cbam_unet']['Best MSE'] * 100
            print(f"  Adding GAN to CBAM (cbam_unet → cbam_gan):")
            print(f"    MSE improvement: {mse_improve:+.2f}%")

        if 'unet_gan' in models and 'cbam_gan' in models:
            mse_improve = (models['unet_gan']['Best MSE'] - models['cbam_gan']['Best MSE']) / models['unet_gan']['Best MSE'] * 100
            print(f"  Adding CBAM to GAN (unet_gan → cbam_gan):")
            print(f"    MSE improvement: {mse_improve:+.2f}%")

        print("=" * 80)

        # 保存CSV
        report_path = os.path.join(PATHS["save_dir"], "ablation_comparison.csv")
        df.to_csv(report_path, index=False)
        print(f"\n[SAVE] Comparison report saved to: {report_path}")

        # 保存JSON
        json_path = os.path.join(PATHS["save_dir"], "ablation_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[SAVE] Detailed results saved to: {json_path}")

        # 生成训练曲线对比数据
        self._save_comparison_curves()

        print("\n" + "=" * 80 + "\n")

    def _save_comparison_curves(self):
        """保存用于绘图的训练曲线对比数据"""
        curves_data = {}

        for model_id, result in self.results.items():
            if result.get("status") != "success":
                continue

            history = result["history"]
            if "epoch" in history and history["epoch"]:
                curves_data[model_id] = {
                    "name": MODEL_REGISTRY[model_id]["name"],
                    "epochs": history["epoch"],
                    "train_loss": history["train_loss"][:len(history["epoch"])],
                    "val_mse": history["val_mse"],
                    "val_psnr": history["val_psnr"],
                    "val_ssim": history["val_ssim"]
                }

        if curves_data:
            curves_path = os.path.join(PATHS["save_dir"], "training_curves_data.json")
            with open(curves_path, 'w') as f:
                json.dump(curves_data, f, indent=2)
            print(f"[SAVE] Training curves data saved to: {curves_path}")


# ========================= 主函数 ========================= #
def main():
    """主函数"""
    import sys

    print("\n" + "=" * 80)
    print("🔬 ABLATION STUDY - 4 Models Comparison")
    print("=" * 80)

    # 可以指定要训练的模型
    if len(sys.argv) > 1:
        model_ids = sys.argv[1:]
        print(f"📋 Training specific models: {model_ids}")
        experiment = AblationExperiment(model_ids=model_ids)
    else:
        print("📋 Training all 4 models: pure_unet, cbam_unet, unet_gan, cbam_gan")
        experiment = AblationExperiment()

    try:
        experiment.run()
        print("\n[SUCCESS] Experiment completed successfully!")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Experiment interrupted by user")

    except Exception as e:
        print(f"\n\n[FAILED] Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()