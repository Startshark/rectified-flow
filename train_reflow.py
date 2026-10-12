import argparse
from datetime import datetime
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
import yaml

from datasets import ReflowDataset
from model import MiniUnet
from rectified_flow import RectifiedFlow


class DualLogger:
    """同时将输出打印到控制台和日志文件"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()


def plot_and_save_loss_curve(loss_list, save_path, epoch=None):
    """绘制并保存损失曲线，包含批次损失与平滑移动平均线"""
    if not loss_list:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Batch Loss', alpha=0.45, color='dodgerblue')

    if len(loss_list) >= 20:
        window = min(50, max(5, len(loss_list) // 20))
        smoothed = np.convolve(loss_list, np.ones(window) / window, mode='valid')
        plt.plot(
            np.arange(window - 1, len(loss_list)),
            smoothed,
            label=f'Moving Avg ({window})',
            color='crimson',
            linewidth=1.8
        )

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    title = 'Reflow Training Loss Curve'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


@torch.no_grad()
def sample_checkpoint_images(
    model,
    rf,
    device,
    save_dir,
    epoch,
    use_cfg=True,
    cfg_scale=5.0,
    step=2,
    fixed_noise=None,
    num_per_class=10
):
    """Reflow 模型采样生成实际图片并保存全景拼图与独立图片（Reflow 2步即可生成高质量图像）"""
    model.eval()
    if use_cfg:
        y = torch.tensor(
            [i for i in range(10) for _ in range(num_per_class)],
            device=device,
            dtype=torch.long
        )
        num_imgs = 10 * num_per_class
        nrow = 10
    else:
        y = None
        num_imgs = 64
        nrow = 8

    if fixed_noise is not None and fixed_noise.shape[0] == num_imgs:
        x_t = fixed_noise.clone().to(device)
    else:
        x_t = torch.randn(num_imgs, 1, 28, 28, device=device)

    dt = 1.0 / step
    for j in range(step):
        t = torch.full((num_imgs,), j * dt, device=device)
        if use_cfg:
            v_uncond = model(x=x_t, t=t)
            v_cond = model(x=x_t, t=t, y=y)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = model(x=x_t, t=t)
        x_t = rf.euler(x_t, v_pred, dt)

    x_t = x_t.clamp(0, 1)

    grid_file = os.path.join(save_dir, f'epoch_{epoch:03d}_grid.png')
    save_image(x_t, grid_file, nrow=nrow)

    indiv_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
    os.makedirs(indiv_dir, exist_ok=True)
    for idx in range(num_imgs):
        label_str = f"_label_{y[idx].item()}" if y is not None else ""
        save_image(x_t[idx], os.path.join(indiv_dir, f'{idx:03d}{label_str}.png'))

    model.train()
    print(f'  [Reflow Checkpoint {epoch}] 采样完成，图像已保存至: {grid_file}')
    return grid_file


def train(config_path: str = './config/train_reflow_config.yaml', output_dir_override: str = None):
    """训练 reflow 模型，输出至时间戳目录"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    base_channels = cfg.get('base_channels', 16)
    epochs = cfg.get('epochs', 10)
    batch_size = cfg.get('batch_size', 128)
    lr_adjust_epoch = cfg.get('lr_adjust_epoch', 50)
    batch_print_interval = cfg.get('batch_print_interval', 100)
    checkpoint_save_interval = cfg.get('checkpoint_save_interval', 1)
    use_cfg = cfg.get('use_cfg', False)
    device = cfg.get('device', 'cuda')

    lr = cfg.get('lr', 1e-5)
    img_root_path = cfg.get('img_root_path', None)
    noise_root_path = cfg.get('noise_root_path', None)
    checkpoint_path = cfg.get('checkpoint_path', None)
    sample_steps = cfg.get('sample_steps', 2)
    sample_cfg_scale = cfg.get('sample_cfg_scale', 5.0)
    sample_num_per_class = cfg.get('sample_num_per_class', 10)

    if device == 'cuda' and not torch.cuda.is_available():
        print('[Warning] CUDA requested but not available on this node. Falling back to cpu.')
        device = 'cpu'

    if output_dir_override:
        output_dir = output_dir_override
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outputs_root = cfg.get('outputs_root', './outputs')
        output_dir = os.path.join(outputs_root, f'reflow_{timestamp}')

    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, 'train_reflow.log')
    logger = DualLogger(log_file_path)
    old_stdout = sys.stdout
    sys.stdout = logger

    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    try:
        print('=' * 60)
        print('Reflow Training Started')
        print(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Output Directory: {os.path.abspath(output_dir)}')
        print(f'Training Config:')
        print(f'  base_channels: {base_channels}')
        print(f'  epochs: {epochs}')
        print(f'  batch_size: {batch_size}')
        print(f'  lr: {lr}')
        print(f'  checkpoint_path: {checkpoint_path}')
        print(f'  device: {device}')
        print('=' * 60)

        transform = Compose([ToTensor()])
        dataset = ReflowDataset(img_root_path=img_root_path, noise_root_path=noise_root_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f'Dataset Loaded: {len(dataset)} samples')

        model = MiniUnet(base_channels)
        model.to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f'Loading pretrained 1-RF checkpoint from {checkpoint_path}...')
            ckpt_data = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt_data['model'])
            print('Checkpoint successfully loaded.')
        else:
            print(f'[Warning] Checkpoint path not found: {checkpoint_path}')

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)
        rf = RectifiedFlow()

        num_sample_imgs = (10 * sample_num_per_class) if use_cfg else 64
        fixed_noise = torch.randn(
            num_sample_imgs, 1, 28, 28,
            generator=torch.Generator().manual_seed(42)
        )

        loss_list = []
        epoch_loss_list = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            batch_count = 0

            for batch, data in enumerate(dataloader):
                x_1 = data['img']
                x_0 = data['noise']
                y = data['label']

                t = torch.rand(x_1.size(0))
                x_t, _ = rf.create_flow(x_1, t, x_0)

                x_t = x_t.to(device)
                x_0 = x_0.to(device)
                x_1 = x_1.to(device)
                t = t.to(device)

                optimizer.zero_grad()

                if use_cfg:
                    x_t = torch.cat([x_t, x_t.clone()], dim=0)
                    t = torch.cat([t, t.clone()], dim=0)
                    y = torch.cat([y, -torch.ones_like(y)], dim=0)
                    x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                    x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                    y = y.to(device)
                else:
                    y = None

                v_pred = model(x=x_t, t=t, y=y)
                loss = rf.mse_loss(v_pred, x_1, x_0)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                loss_list.append(loss_val)
                epoch_loss += loss_val
                batch_count += 1

                if batch % batch_print_interval == 0:
                    print(f'[Epoch {epoch}/{epochs}] [Batch {batch}/{len(dataloader)}] loss: {loss_val:.6f}')

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            epoch_loss_list.append(avg_epoch_loss)

            if epoch % checkpoint_save_interval == 0 or epoch == epochs:
                save_dict = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    loss_list=loss_list,
                    epoch_loss_list=epoch_loss_list,
                    config=cfg
                )
                ckpt_file = os.path.join(ckpt_dir, f'miniunet_{epoch}.pth')
                torch.save(save_dict, ckpt_file)

                loss_curve_file = os.path.join(output_dir, 'loss_curve.png')
                plot_and_save_loss_curve(loss_list, loss_curve_file, epoch=epoch)

                with open(os.path.join(output_dir, 'loss_history.json'), 'w', encoding='utf-8') as f:
                    json.dump({'loss_list': loss_list, 'epoch_loss_list': epoch_loss_list}, f)

                sample_checkpoint_images(
                    model=model,
                    rf=rf,
                    device=device,
                    save_dir=samples_dir,
                    epoch=epoch,
                    use_cfg=use_cfg,
                    cfg_scale=sample_cfg_scale,
                    step=sample_steps,
                    fixed_noise=fixed_noise,
                    num_per_class=sample_num_per_class
                )

        torch.save(save_dict, os.path.join(ckpt_dir, 'miniunet_final.pth'))
        plot_and_save_loss_curve(loss_list, os.path.join(output_dir, 'loss_curve.png'))
        print(f'Reflow training complete! Outputs saved to: {os.path.abspath(output_dir)}')

    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Reflow model with logging and sampling')
    parser.add_argument('--config', type=str, default='./config/train_reflow_config.yaml', help='Config path')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory')
    args = parser.parse_args()

    train(config_path=args.config, output_dir_override=args.output_dir)
