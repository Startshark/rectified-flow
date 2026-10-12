"""可插拔的训练策略模块 (Pluggable Training Strategies)

三个可定制维度:
  1. Time Sampler   — 如何采样 t ∈ (0, 1)
  2. Loss Function  — 用什么损失函数
  3. Loss Weighter  — 不同 t 的 loss 如何加权

使用方式:
    # 从 YAML 配置构建
    sampler  = build_time_sampler({'name': 'logit_normal', 'params': {'loc': 0.0, 'scale': 1.0}})
    loss_fn  = build_loss_fn({'name': 'mse'})
    weighter = build_loss_weighter({'name': 'snr', 'params': {'gamma': 5.0}})

    # 自定义并注册
    @register_time_sampler('my_sampler')
    class MySampler:
        def __call__(self, batch_size, device='cpu'):
            return torch.rand(batch_size, device=device) ** 0.5
"""

import math
import torch
import torch.nn.functional as F


# ============================================================
# 工具函数
# ============================================================

def _reduce_spatial(x):
    """将 [B, C, H, W] 沿空间维度取均值，得到 [B]"""
    return x.mean(dim=list(range(1, x.ndim)))


# ============================================================
# Time Samplers — 签名: (batch_size, device) -> Tensor[B]
# ============================================================

class UniformTimeSampler:
    """标准均匀采样 t ~ U(0, 1)"""

    def __call__(self, batch_size, device='cpu'):
        return torch.rand(batch_size, device=device)

    def __repr__(self):
        return 'UniformTimeSampler()'


class LogitNormalTimeSampler:
    """Logit-Normal 采样 (SD3 / Flux 论文推荐)

    在 logit 空间做正态采样后 sigmoid 映射回 [0,1]，
    偏向 t≈0 和 t≈1 附近的困难区域。

    Args:
        loc:   logit 空间正态分布的均值
        scale: logit 空间正态分布的标准差
    """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def __call__(self, batch_size, device='cpu'):
        u = torch.randn(batch_size, device=device) * self.scale + self.loc
        return torch.sigmoid(u)

    def __repr__(self):
        return f'LogitNormalTimeSampler(loc={self.loc}, scale={self.scale})'


class BetaTimeSampler:
    """Beta 分布采样

    alpha < 1 且 beta < 1 时偏向两端 (U型)，
    alpha > 1 且 beta > 1 时偏向中间 (倒U型)。

    Args:
        alpha: Beta 分布的 α 参数
        beta:  Beta 分布的 β 参数
    """

    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.dist = torch.distributions.Beta(alpha, beta)

    def __call__(self, batch_size, device='cpu'):
        return self.dist.sample((batch_size,)).to(device)

    def __repr__(self):
        return f'BetaTimeSampler(alpha={self.alpha}, beta={self.beta})'


class CosineTimeSampler:
    """Cosine 调度采样，偏向 t≈0 (噪声端)"""

    def __call__(self, batch_size, device='cpu'):
        u = torch.rand(batch_size, device=device)
        return 1 - torch.cos(u * math.pi / 2)

    def __repr__(self):
        return 'CosineTimeSampler()'


class StratifiedTimeSampler:
    """分层采样

    把 [0,1] 等分为 batch_size 个区间，
    每个区间内均匀随机采一个点，降低方差。
    """

    def __call__(self, batch_size, device='cpu'):
        offsets = torch.arange(batch_size, dtype=torch.float32, device=device)
        t = (offsets + torch.rand(batch_size, device=device)) / batch_size
        return t

    def __repr__(self):
        return 'StratifiedTimeSampler()'


# ============================================================
# Loss Functions — 签名: (pred, target) -> Tensor[B] (per-sample)
# ============================================================

class MSELoss:
    """标准 MSE Loss (per-sample)"""

    def __call__(self, pred, target):
        return _reduce_spatial((pred - target) ** 2)

    def __repr__(self):
        return 'MSELoss()'


class HuberLoss:
    """Huber Loss，对 outlier 更鲁棒

    Args:
        delta: Huber 损失的 δ 阈值
    """

    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, pred, target):
        # F.huber_loss reduction='none' 返回与输入同维度
        loss = F.huber_loss(pred, target, reduction='none', delta=self.delta)
        return _reduce_spatial(loss)

    def __repr__(self):
        return f'HuberLoss(delta={self.delta})'


class PseudoHuberLoss:
    """Pseudo-Huber Loss (Karras et al. 推荐, 用于 flow matching)

    L = sqrt((pred - target)^2 + c^2) - c

    Args:
        c: 平滑常数，越小越接近 L1
    """

    def __init__(self, c=0.00054):
        self.c = c

    def __call__(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.c ** 2) - self.c
        return _reduce_spatial(loss)

    def __repr__(self):
        return f'PseudoHuberLoss(c={self.c})'


class CharbonnierLoss:
    """Charbonnier Loss，近似 L1 但处处可微

    L = sqrt((pred - target)^2 + eps^2)

    Args:
        eps: 平滑常数
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, pred, target):
        loss = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        return _reduce_spatial(loss)

    def __repr__(self):
        return f'CharbonnierLoss(eps={self.eps})'


class L1Loss:
    """L1 / MAE Loss (per-sample)"""

    def __call__(self, pred, target):
        return _reduce_spatial(torch.abs(pred - target))

    def __repr__(self):
        return 'L1Loss()'


# ============================================================
# Loss Weighters — 签名: (t: Tensor[B]) -> Tensor[B] (权重)
# ============================================================

class UniformWeighter:
    """所有 t 权重相同（默认行为）"""

    def __call__(self, t):
        return torch.ones_like(t)

    def __repr__(self):
        return 'UniformWeighter()'


class SNRWeighter:
    """Min-SNR-γ 加权 (Hang et al.)

    在 rectified flow 中 SNR(t) ≈ t / (1 - t)，
    权重 = min(SNR, γ) / SNR

    Args:
        gamma: SNR 截断阈值
    """

    def __init__(self, gamma=5.0):
        self.gamma = gamma

    def __call__(self, t):
        snr = t / (1 - t + 1e-8)
        return torch.clamp(snr, max=self.gamma) / (snr + 1e-8)

    def __repr__(self):
        return f'SNRWeighter(gamma={self.gamma})'


class CosineWeighter:
    """Cosine 加权，对 t≈0.5 附近的中间区域给予更多权重"""

    def __call__(self, t):
        return 1 - torch.cos(t * math.pi)

    def __repr__(self):
        return 'CosineWeighter()'


class LinearWeighter:
    """线性加权，t 越大（越接近数据端）权重越高"""

    def __call__(self, t):
        return t + 0.1  # 加小常数防止权重为 0

    def __repr__(self):
        return 'LinearWeighter()'


class InverseWeighter:
    """逆向线性加权，t 越小（噪声端）权重越高"""

    def __call__(self, t):
        return 1 - t + 0.1

    def __repr__(self):
        return 'InverseWeighter()'


class LognormalWeighter:
    """Lognormal density weighting (类似 EDM2)

    基于 logit(t) 的正态密度分配权重。

    Args:
        loc:   logit 空间正态分布的均值
        scale: logit 空间正态分布的标准差
    """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def __call__(self, t):
        logit_t = torch.log(t / (1 - t + 1e-8) + 1e-8)
        log_p = -0.5 * ((logit_t - self.loc) / self.scale) ** 2
        return torch.exp(log_p)

    def __repr__(self):
        return f'LognormalWeighter(loc={self.loc}, scale={self.scale})'


# ============================================================
# Registries
# ============================================================

TIME_SAMPLERS = {
    'uniform': UniformTimeSampler,
    'logit_normal': LogitNormalTimeSampler,
    'beta': BetaTimeSampler,
    'cosine': CosineTimeSampler,
    'stratified': StratifiedTimeSampler,
}

LOSS_FUNCTIONS = {
    'mse': MSELoss,
    'huber': HuberLoss,
    'pseudo_huber': PseudoHuberLoss,
    'charbonnier': CharbonnierLoss,
    'l1': L1Loss,
}

LOSS_WEIGHTERS = {
    'uniform': UniformWeighter,
    'snr': SNRWeighter,
    'cosine': CosineWeighter,
    'linear': LinearWeighter,
    'inverse': InverseWeighter,
    'lognormal': LognormalWeighter,
}


# ============================================================
# 注册装饰器 — 用户自定义策略
# ============================================================

def register_time_sampler(name):
    """注册自定义 Time Sampler

    用法:
        @register_time_sampler('my_sampler')
        class MySampler:
            def __call__(self, batch_size, device='cpu'):
                ...
    """
    def decorator(cls):
        TIME_SAMPLERS[name] = cls
        return cls
    return decorator


def register_loss_fn(name):
    """注册自定义 Loss Function

    用法:
        @register_loss_fn('my_loss')
        class MyLoss:
            def __call__(self, pred, target):
                # 返回 per-sample loss [B]
                ...
    """
    def decorator(cls):
        LOSS_FUNCTIONS[name] = cls
        return cls
    return decorator


def register_loss_weighter(name):
    """注册自定义 Loss Weighter

    用法:
        @register_loss_weighter('my_weighter')
        class MyWeighter:
            def __call__(self, t):
                # 返回权重 [B]
                ...
    """
    def decorator(cls):
        LOSS_WEIGHTERS[name] = cls
        return cls
    return decorator


# ============================================================
# Build 函数 — 从 YAML 配置字典构建策略对象
# ============================================================

def build_time_sampler(cfg=None):
    """从配置字典构建 Time Sampler

    Args:
        cfg: {'name': 'logit_normal', 'params': {'loc': 0.0, 'scale': 1.0}}
             若为 None 或空，返回 UniformTimeSampler
    """
    if cfg is None:
        cfg = {}
    name = cfg.get('name', 'uniform')
    params = cfg.get('params', {})
    if name not in TIME_SAMPLERS:
        raise ValueError(
            f"Unknown time sampler '{name}'. "
            f"Available: {list(TIME_SAMPLERS.keys())}. "
            f"Use @register_time_sampler('{name}') to register a custom one."
        )
    return TIME_SAMPLERS[name](**params)


def build_loss_fn(cfg=None):
    """从配置字典构建 Loss Function

    Args:
        cfg: {'name': 'mse', 'params': {}}
             若为 None 或空，返回 MSELoss
    """
    if cfg is None:
        cfg = {}
    name = cfg.get('name', 'mse')
    params = cfg.get('params', {})
    if name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Available: {list(LOSS_FUNCTIONS.keys())}. "
            f"Use @register_loss_fn('{name}') to register a custom one."
        )
    return LOSS_FUNCTIONS[name](**params)


def build_loss_weighter(cfg=None):
    """从配置字典构建 Loss Weighter

    Args:
        cfg: {'name': 'snr', 'params': {'gamma': 5.0}}
             若为 None 或空，返回 UniformWeighter
    """
    if cfg is None:
        cfg = {}
    name = cfg.get('name', 'uniform')
    params = cfg.get('params', {})
    if name not in LOSS_WEIGHTERS:
        raise ValueError(
            f"Unknown loss weighter '{name}'. "
            f"Available: {list(LOSS_WEIGHTERS.keys())}. "
            f"Use @register_loss_weighter('{name}') to register a custom one."
        )
    return LOSS_WEIGHTERS[name](**params)

