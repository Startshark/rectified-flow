import torch
import torch.nn.functional as F

from training_strategies import (
    UniformTimeSampler, MSELoss, UniformWeighter,
)

# 老司机开车理论->三要素：路线、车、司机


class RectifiedFlow:
    def __init__(self, time_sampler=None, loss_fn=None, loss_weighter=None):
        """初始化 RectifiedFlow

        Args:
            time_sampler: 时间采样策略，callable(batch_size, device) -> Tensor[B]
            loss_fn:      损失函数策略，callable(pred, target) -> Tensor[B] (per-sample)
            loss_weighter: 损失加权策略，callable(t) -> Tensor[B] (权重)
        """
        self.time_sampler = time_sampler or UniformTimeSampler()
        self.loss_fn = loss_fn or MSELoss()
        self.loss_weighter = loss_weighter or UniformWeighter()

    # 采样时间 t
    def sample_t(self, batch_size, device='cpu'):
        """采样时间 t ∈ (ε, 1-ε)，避免边界数值问题

        Args:
            batch_size: 批次大小
            device: 设备
        Returns:
            t: 采样的时间，维度为 [B]
        """
        t = self.time_sampler(batch_size, device)
        return t.clamp(1e-5, 1 - 1e-5)

    # 车：图像生成一个迭代公式 ODE f(t+dt) = f(t) + dt*f'(t)
    def euler(self, x_t, v, dt):
        """ 使用欧拉方法计算下一个时间步长的值
            
        Args:
            x_t: 当前的值，维度为 [B, C, H, W]
            v: 当前的速度，维度为 [B, C, H, W]
            dt: 时间步长
        """
        x_t = x_t + v * dt

        return x_t

    # 路线
    # v1.2: reflow增加x_0的输入
    def create_flow(self, x_1, t, x_0=None):
        """ 使用x_t = t * x_1 + (1 - t) * x_0公式构建x_0到x_1的流

            X_1是原始图像 X_0是噪声图像（服从标准高斯分布）
            
        Args:
            x_1: 原始图像，维度为 [B, C, H, W]
            t: 一个标量，表示时间，时间范围为 [0, 1]，维度为 [B]
            x_0: 噪声图像，维度为 [B, C, H, W]，默认值为None
            
        Returns:
            x_t: 在时间t的图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        
        """

        # 需要一个x0，x0服从高斯噪声
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=x_1.dtype, device=x_1.device)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        t = t[:, None, None, None]  # [B, 1, 1, 1]

        # 获得xt的值
        x_t = t * x_1 + (1 - t) * x_0

        return x_t, x_0

    # 司机：可插拔的损失计算
    def compute_loss(self, v_pred, x_1, x_0, t):
        """计算加权损失（支持自定义损失函数和加权策略）

        Args:
            v_pred: 模型预测的速度，维度为 [B, C, H, W]
            x_1: 原始图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
            t: 采样时间，维度为 [B]
        Returns:
            loss: 标量损失值
        """
        target = x_1 - x_0  # flow matching target: velocity = x_1 - x_0

        # 计算 per-sample loss [B]
        per_sample_loss = self.loss_fn(v_pred, target)

        # 计算 per-sample weight [B]
        weights = self.loss_weighter(t)

        # 加权平均
        return (weights * per_sample_loss).mean()

    # 保留原有 mse_loss 做向后兼容
    def mse_loss(self, v, x_1, x_0):
        """ 计算RectifiedFlow的损失函数（向后兼容）
        L = MSE(x_1 - x_0 - v(t))  匀速直线运动

        Args:
            v: 速度，维度为 [B, C, H, W]
            x_1: 原始图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        """

        # 求loss函数，是一个MSE，最后维度是[B]

        loss = F.mse_loss(x_1 - x_0, v)
        # loss = torch.mean((x_1 - x_0 - v)**2)

        return loss


if __name__ == '__main__':
    # 时间越大，越是接近原始图像

    rf = RectifiedFlow()

    x_t = rf.create_flow(torch.ones(2, 3, 4, 4), 0.999)

    print(x_t)
