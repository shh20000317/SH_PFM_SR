"""
优化的消融实验模型注册系统
4个模型的清晰对比:
1. Pure-UNet: 基础UNet + L1损失 (最简单基准)
2. CBAM-UNet: CBAM注意力UNet + L1损失 (验证注意力作用)
3. UNet-GAN: 基础UNet + GAN + 复杂损失 (验证GAN作用)
4. CBAM-GAN: CBAM + GAN + 复杂损失 (完整模型)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import functools

# ========================= 模型配置注册表 ========================= #
MODEL_REGISTRY = {
    "pure_unet": {
        "name": "Pure-UNet",
        "description": "Baseline: Basic UNet + L1 Loss Only",
        "use_gan": False,
        "use_attention": False,
        "loss_type": "simple",
        "model_file": "model_unet_gan",
        "attention_config": None,
        "color": "blue",
        "order": 1,
        "purpose": "基准模型 - 验证基础架构性能"
    },
    "cbam_unet": {
        "name": "CBAM-UNet",
        "description": "CBAM Attention + L1 Loss (No GAN)",
        "use_gan": False,
        "use_attention": True,
        "loss_type": "simple",
        "model_file": "model_cbam_gan",
        "attention_config": {"attention_type": "simple", "use_multi_scale": False},
        "color": "green",
        "order": 2,
        "purpose": "注意力消融 - 验证CBAM注意力机制的作用"
    },
    "unet_gan": {
        "name": "UNet-GAN",
        "description": "Basic UNet + GAN + Complex Loss",
        "use_gan": True,
        "use_attention": False,
        "loss_type": "complex",
        "model_file": "model_unet_gan",
        "attention_config": None,
        "color": "orange",
        "order": 3,
        "purpose": "GAN消融 - 验证对抗训练和复杂损失的作用"
    },
    "cbam_gan": {
        "name": "CBAM-GAN (Full Model)",
        "description": "CBAM + GAN + Complex Loss (Complete)",
        "use_gan": True,
        "use_attention": True,
        "loss_type": "complex",
        "model_file": "model_cbam_gan",
        "attention_config": {"attention_type": "simple", "use_multi_scale": False},
        "color": "purple",
        "order": 4,
        "purpose": "完整模型 - 结合所有改进"
    }
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """获取模型配置"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def create_generator(model_name: str, input_nc: int, output_nc: int,
                     ngf: int = 32, norm_layer=None) -> nn.Module:
    """根据模型名称创建生成器"""
    config = get_model_config(model_name)

    if norm_layer is None:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)

    if config["model_file"] == "model_unet_gan":
        from model_unet_gan import UnetGenerator
        model = UnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=4,
            ngf=ngf,
            norm_layer=norm_layer
        )
    elif config["model_file"] == "model_cbam_gan":
        from model_cbam_gan import UnetGenerator
        att_config = config["attention_config"] or {}
        model = UnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=4,
            ngf=ngf,
            norm_layer=norm_layer,
            **att_config
        )
    else:
        raise ValueError(f"Unknown model_file: {config['model_file']}")

    return model


def create_discriminator(ndf: int = 32, input_nc: int = 3,
                         n_layers: int = 3, norm_layer=None) -> nn.Module:
    """创建PatchGAN判别器"""
    if norm_layer is None:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)

    try:
        from model_cbam_gan import NLayerDiscriminator
    except:
        from model_unet_gan import NLayerDiscriminator

    return NLayerDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_layer=norm_layer
    )


def create_loss_function(model_name: str, discriminator: nn.Module = None) -> nn.Module:
    """根据模型配置创建损失函数"""
    config = get_model_config(model_name)

    if config["loss_type"] == "simple":
        return SimpleL1Loss()
    elif config["loss_type"] == "complex":
        if discriminator is None:
            raise ValueError(f"Discriminator required for {model_name}")
        from loss_function import HydrologicalGANLoss
        return HydrologicalGANLoss(
            discriminator=discriminator,
            loss_type="lsgan",
            g_rec_weight=50.0,
            edge_weight=0.5,
            empty_weight=0.5,
            perc_weight=0.1,
            adv_weight=1.0,
            lambda_gp=10.0
        )
    else:
        raise ValueError(f"Unknown loss_type: {config['loss_type']}")


class SimpleL1Loss(nn.Module):
    """简单的L1重建损失"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def compute_generator_loss(self, fake_pred: torch.Tensor, real_img: torch.Tensor,
                               fake_img: torch.Tensor, valid_mask: torch.Tensor):
        """计算生成器损失"""
        masked_pred = fake_img * valid_mask
        masked_real = real_img * valid_mask
        loss = self.l1(masked_pred, masked_real)

        return {
            'total': loss,
            'rec': loss,
            'adv': torch.tensor(0.0, device=loss.device),
            'empty': torch.tensor(0.0, device=loss.device),
            'edge': torch.tensor(0.0, device=loss.device),
            'perc': torch.tensor(0.0, device=loss.device)
        }

    def compute_discriminator_loss(self, *args, **kwargs):
        """占位函数"""
        return {
            'total': torch.tensor(0.0),
            'main': torch.tensor(0.0),
            'gp': torch.tensor(0.0),
            'real_score': 0.0,
            'fake_score': 0.0
        }


def print_model_summary():
    """打印模型对比表格"""
    print("\n" + "=" * 100)
    print("🔬 ABLATION STUDY: 4-Model Comparison")
    print("=" * 100)

    sorted_models = sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]['order'])

    print(f"{'#':<4} {'Model ID':<15} {'Name':<25} {'Attention':<12} {'GAN':<6} {'Loss':<10}")
    print("-" * 100)

    for model_id, config in sorted_models:
        order = config['order']
        att = "✓" if config['use_attention'] else "✗"
        gan = "✓" if config['use_gan'] else "✗"
        loss = config['loss_type']
        print(f"{order:<4} {model_id:<15} {config['name']:<25} {att:<12} {gan:<6} {loss:<10}")

    print("=" * 100)
    print("\n📊 Ablation Logic:")
    print("-" * 100)
    for model_id, config in sorted_models:
        print(f"  {config['order']}. [{model_id:>12}] {config['name']:<25}")
        print(f"      → {config['purpose']}")

    print("\n🎯 Comparison Strategy:")
    print("-" * 100)
    print("  Step 1→2: pure_unet → cbam_unet       | Isolate CBAM attention effect")
    print("  Step 1→3: pure_unet → unet_gan        | Isolate GAN + complex loss effect")
    print("  Step 2→4: cbam_unet → cbam_gan        | Add GAN to attention model")
    print("  Step 3→4: unet_gan → cbam_gan         | Add attention to GAN model")
    print("  Final:    Compare all to identify best combination")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print_model_summary()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")

    for model_id in MODEL_REGISTRY.keys():
        print(f"{'=' * 80}")
        print(f"Testing: {model_id}")
        print(f"{'=' * 80}")

        config = get_model_config(model_id)

        try:
            G = create_generator(model_id, input_nc=3, output_nc=3, ngf=32).to(device)
            g_params = sum(p.numel() for p in G.parameters())
            print(f"  ✓ Generator created: {g_params:,} parameters")

            if config['use_gan']:
                D = create_discriminator(ndf=32, input_nc=3).to(device)
                d_params = sum(p.numel() for p in D.parameters())
                print(f"  ✓ Discriminator created: {d_params:,} parameters")
                criterion = create_loss_function(model_id, D)
            else:
                criterion = create_loss_function(model_id)

            print(f"  ✓ Loss function: {criterion.__class__.__name__}")

            x = torch.randn(1, 3, 256, 256).to(device)
            with torch.no_grad():
                output = G(x)
                print(f"  ✓ Forward pass successful: {output.shape}")

            print(f"  ✅ {model_id} test PASSED!\n")

        except Exception as e:
            print(f"  ❌ {model_id} test FAILED: {e}\n")
            import traceback
            traceback.print_exc()