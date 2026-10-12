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
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
import yaml

from model import MiniUnet
from rectified_flow import RectifiedFlow
from training_strategies import build_time_sampler, build_loss_fn, build_loss_weighter


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
    title = 'Rectified Flow Loss Curve'
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
    step=20,
    fixed_noise=None,
    num_per_class=1
):
    """在每个 checkpoint 时采样生成实际图片并保存全景拼图与独立图片
    默认 num_per_class=1，条件生成时 0~9 每个数字采样 1 张，共 10 张图片
    """
    model.eval()
    if use_cfg:
        # 条件生成：0-9 每个数字采样 num_per_class 张（默认各1张，总计10张）
        y = torch.tensor(
            [i for i in range(10) for _ in range(num_per_class)],
            device=device,
            dtype=torch.long
        )
        num_imgs = 10 * num_per_class
        nrow = 10
    else:
        # 无条件生成：采样 10 张
        y = None
        num_imgs = 10
        nrow = 10

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

    # 1. 保存全景拼图 grid (1行10列)
    grid_file = os.path.join(save_dir, f'epoch_{epoch:03d}_grid.png')
    save_image(x_t, grid_file, nrow=nrow)

    # 2. 保存各张独立采样图片
    indiv_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
    os.makedirs(indiv_dir, exist_ok=True)
    for idx in range(num_imgs):
        label_str = f"_label_{y[idx].item()}" if y is not None else ""
        save_image(x_t[idx], os.path.join(indiv_dir, f'{idx:02d}{label_str}.png'))

    model.train()
    print(f'  [Checkpoint {epoch}] 采样完成（共 {num_imgs} 张），拼图已保存至: {grid_file}')
    return grid_file


def train(config_path: str = './config/train_config.yaml', output_dir_override: str = None, wandb_override: bool = None):
    """训练 flow matching (rectified flow) 模型，并统一输出到时间戳目录

    Args:
        config_path (str): yaml配置文件路径
        output_dir_override (str, optional): 自定义输出目录，若为 None 则自动创建 outputs/<timestamp>
        wandb_override (bool, optional): 显式开启或关闭 wandb 上传
    """
    # 读取yaml配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 解析超参数
    base_channels = cfg.get('base_channels', 16)
    epochs = cfg.get('epochs', 10)
    batch_size = cfg.get('batch_size', 128)
    lr_adjust_epoch = cfg.get('lr_adjust_epoch', 50)
    batch_print_interval = cfg.get('batch_print_interval', 100)
    checkpoint_save_interval = cfg.get('checkpoint_save_interval', 1)
    use_cfg = cfg.get('use_cfg', False)
    device = cfg.get('device', 'cuda')
    sample_steps = cfg.get('sample_steps', 20)
    sample_cfg_scale = cfg.get('sample_cfg_scale', 5.0)
    sample_num_per_class = cfg.get('sample_num_per_class', 1)  # 默认每个数字采样1张，共10张

    # 是否启用 wandb
    use_wandb = cfg.get('use_wandb', False)
    if wandb_override is not None:
        use_wandb = wandb_override
    wandb_project = cfg.get('wandb_project', 'rectified-flow')
    wandb_entity = cfg.get('wandb_entity', None)
    wandb_run_name = cfg.get('wandb_run_name', None)

    if device == 'cuda' and not torch.cuda.is_available():
        print('[Warning] CUDA requested but not available on this node. Falling back to cpu.')
        device = 'cpu'

    # 创建 outputs 下带时间戳的文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir_override:
        output_dir = output_dir_override
    else:
        outputs_root = cfg.get('outputs_root', './outputs')
        output_dir = os.path.join(outputs_root, timestamp)

    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # 配置日志重定向：同时写控制台和 train.log
    log_file_path = os.path.join(output_dir, 'train.log')
    logger = DualLogger(log_file_path)
    old_stdout = sys.stdout
    sys.stdout = logger

    # 保存本次训练的 config 副本，方便回溯复现
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    # 初始化 wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            # 自动探测 entity，防止因属于高校/企业团队组织导致 entity not specified 错误
            if not wandb_entity:
                try:
                    _api = wandb.Api()
                    if getattr(_api.viewer, 'teams', None) and len(_api.viewer.teams) > 0:
                        wandb_entity = _api.viewer.teams[0]
                    elif _api.default_entity:
                        wandb_entity = _api.default_entity
                    else:
                        wandb_entity = _api.viewer.username
                except Exception:
                    pass

            run_name = wandb_run_name or f"rf-{timestamp}"
            wandb_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=run_name,
                config=cfg,
                dir=output_dir
            )
            print(f'W&B logging enabled! Run URL: {wandb_run.url}')
        except Exception as e:
            print(f'[Warning] Failed to initialize wandb ({e}). Continuing without wandb.')
            use_wandb = False
            wandb_run = None

    try:
        print('=' * 60)
        print('Rectified Flow Training Started')
        print(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Output Directory: {os.path.abspath(output_dir)}')
        print(f'Checkpoints Directory: {os.path.abspath(ckpt_dir)}')
        print(f'Samples Directory: {os.path.abspath(samples_dir)}')
        print('Training Config:')
        print(f'  base_channels: {base_channels}')
        print(f'  epochs: {epochs}')
        print(f'  batch_size: {batch_size}')
        print(f'  lr_adjust_epoch: {lr_adjust_epoch}')
        print(f'  batch_print_interval: {batch_print_interval}')
        print(f'  checkpoint_save_interval: {checkpoint_save_interval}')
        print(f'  use_cfg: {use_cfg}')
        print(f'  device: {device}')
        print(f'  sample_steps: {sample_steps}')
        print(f'  sample_cfg_scale: {sample_cfg_scale}')
        print(f'  sample_num_per_class: {sample_num_per_class} (Total {10 * sample_num_per_class if use_cfg else 10} images)')
        print(f'  use_wandb: {use_wandb}')
        print('=' * 60)

        # 数据集加载
        transform = Compose([ToTensor()])
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f'Dataset Loaded: {len(dataset)} samples, {len(dataloader)} batches per epoch.')

        # 模型加载
        model = MiniUnet(base_channels)
        model.to(device)

        # 优化器与学习率调整
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
        scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

        # RF 算法 + 训练策略
        strategy_cfg = cfg.get('training_strategy', {})
        time_sampler = build_time_sampler(strategy_cfg.get('time_sampler'))
        loss_fn = build_loss_fn(strategy_cfg.get('loss_fn'))
        loss_weighter = build_loss_weighter(strategy_cfg.get('loss_weighter'))
        rf = RectifiedFlow(time_sampler=time_sampler, loss_fn=loss_fn, loss_weighter=loss_weighter)
        print(f'  time_sampler:  {rf.time_sampler}')
        print(f'  loss_fn:       {rf.loss_fn}')
        print(f'  loss_weighter: {rf.loss_weighter}')

        # 固定初始高斯噪声，方便在不同 checkpoint 之间直观对比同一噪声生成质量的变化
        num_sample_imgs = (10 * sample_num_per_class) if use_cfg else 10
        fixed_noise = torch.randn(
            num_sample_imgs, 1, 28, 28,
            generator=torch.Generator().manual_seed(42)
        )

        loss_list = []
        epoch_loss_list = []
        global_step = 0

        # 训练前先采样一次未训练的初态（epoch 0 前）作为 baseline 对照
        print('\n--> 初始采样 (未经训练的模型初态)...')
        init_grid = sample_checkpoint_images(
            model=model,
            rf=rf,
            device=device,
            save_dir=samples_dir,
            epoch=0,
            use_cfg=use_cfg,
            cfg_scale=sample_cfg_scale,
            step=sample_steps,
            fixed_noise=fixed_noise,
            num_per_class=sample_num_per_class
        )
        if use_wandb and wandb_run:
            import wandb
            wandb.log({'samples/grid': wandb.Image(init_grid, caption='Epoch 0 (Initial)'), 'epoch': 0, 'iteration': 0})

        # 训练主循环
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            batch_count = 0

            for batch, data in enumerate(dataloader):
                x_1, y = data  # x_1 原始图像，y 是标签
                t = rf.sample_t(x_1.size(0))

                x_t, x_0 = rf.create_flow(x_1, t)

                x_t = x_t.to(device)
                x_0 = x_0.to(device)
                x_1 = x_1.to(device)
                t = t.to(device)

                optimizer.zero_grad()

                if use_cfg:
                    # 50% 概率做无条件拼接
                    x_t = torch.cat([x_t, x_t.clone()], dim=0)
                    t = torch.cat([t, t.clone()], dim=0)
                    y = torch.cat([y, -torch.ones_like(y)], dim=0)
                    x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                    x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                    y = y.to(device)
                else:
                    y = None

                v_pred = model(x=x_t, t=t, y=y)
                loss = rf.compute_loss(v_pred, x_1, x_0, t)

                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                loss_list.append(loss_val)
                epoch_loss += loss_val
                batch_count += 1
                global_step += 1

                if use_wandb and wandb_run and (global_step % 10 == 0):
                    import wandb
                    wandb.log({
                        'train/batch_loss': loss_val,
                        'train/lr': scheduler.get_last_lr()[0],
                        'iteration': global_step
                    })

                if batch % batch_print_interval == 0:
                    print(f'[Epoch {epoch}/{epochs}] [Batch {batch}/{len(dataloader)}] loss: {loss_val:.6f}')

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            epoch_loss_list.append(avg_epoch_loss)
            print(f'>>> [Epoch {epoch} finished] Average Loss: {avg_epoch_loss:.6f}')

            if use_wandb and wandb_run:
                import wandb
                wandb.log({
                    'train/epoch_loss': avg_epoch_loss,
                    'epoch': epoch,
                    'iteration': global_step
                })

            # 达到保存间隔或最后一轮时保存模型、绘制曲线、采样图像
            if epoch % checkpoint_save_interval == 0 or epoch == epochs:
                print(f'\n--> 保存 Checkpoint (Epoch {epoch})...')
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
                print(f'  权重已保存: {ckpt_file}')

                # 记录并保存损失曲线
                loss_curve_file = os.path.join(output_dir, 'loss_curve.png')
                plot_and_save_loss_curve(loss_list, loss_curve_file, epoch=epoch)
                print(f'  损失曲线已更新: {loss_curve_file}')

                # 保存 loss 数据 json
                loss_json_file = os.path.join(output_dir, 'loss_history.json')
                with open(loss_json_file, 'w', encoding='utf-8') as f:
                    json.dump({'loss_list': loss_list, 'epoch_loss_list': epoch_loss_list}, f)

                # 采样实际图片 (10 张)
                grid_file = sample_checkpoint_images(
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

                if use_wandb and wandb_run:
                    import wandb
                    wandb.log({
                        'samples/grid': wandb.Image(grid_file, caption=f"Epoch {epoch}"),
                        'epoch': epoch,
                        'iteration': global_step
                    })
                print()

        # 保存最终权重
        final_ckpt = os.path.join(ckpt_dir, 'miniunet_final.pth')
        torch.save(save_dict, final_ckpt)
        loss_curve_file = os.path.join(output_dir, 'loss_curve.png')
        plot_and_save_loss_curve(loss_list, loss_curve_file)

        if use_wandb and wandb_run:
            import wandb
            if os.path.exists(loss_curve_file):
                wandb.log({'charts/loss_curve': wandb.Image(loss_curve_file)})
            wandb.finish()

        print(f'Final model saved to: {final_ckpt}')
        print(f'All outputs successfully saved to: {os.path.abspath(output_dir)}')

    finally:
        sys.stdout = old_stdout
        logger.close()
        if use_wandb and wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Rectified Flow with auto logging, sampling and optional wandb')
    parser.add_argument('--config', type=str, default='./config/train_config.yaml', help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory (default: outputs/<timestamp>)')
    parser.add_argument('--wandb', action='store_true', default=None, help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()

    wandb_override = None
    if args.wandb:
        wandb_override = True
    elif args.no_wandb:
        wandb_override = False

    train(config_path=args.config, output_dir_override=args.output_dir, wandb_override=wandb_override)
