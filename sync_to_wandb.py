import os
import sys
import json
import yaml
import glob
import wandb

def sync_run(run_dir=None):
    if not run_dir:
        # 默认同步最新的 outputs 文件夹
        runs = sorted(glob.glob('./outputs/*'))
        runs = [r for r in runs if os.path.isdir(r)]
        if not runs:
            print("未找到任何 outputs 运行目录！")
            return
        run_dir = runs[-1]

    run_dir = os.path.abspath(run_dir)
    print(f"正在同步运行记录: {run_dir} 到 Weights & Biases...")

    # 读取配置
    cfg_file = os.path.join(run_dir, 'config.yaml')
    cfg = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader) or {}

    project = cfg.get('wandb_project', 'rectified-flow')
    entity = cfg.get('wandb_entity', 'ruh_cmu-carnegie-mellon-university')
    run_name = os.path.basename(run_dir)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=cfg,
        dir=run_dir
    )

    # 同步 loss 历史
    loss_json = os.path.join(run_dir, 'loss_history.json')
    if os.path.exists(loss_json):
        with open(loss_json, 'r', encoding='utf-8') as f:
            loss_data = json.load(f)
            loss_list = loss_data.get('loss_list', [])
            epoch_loss_list = loss_data.get('epoch_loss_list', [])

            # 稀疏采样记录 batch loss（每隔10个batch记录一次，避免请求过多）
            for step, val in enumerate(loss_list):
                if step % 10 == 0 or step == len(loss_list) - 1:
                    wandb.log({'train/batch_loss': val, 'iteration': step})

            for ep, val in enumerate(epoch_loss_list, start=1):
                wandb.log({'train/epoch_loss': val, 'epoch': ep})

    # 同步采样的图片
    samples_dir = os.path.join(run_dir, 'samples')
    if os.path.exists(samples_dir):
        grid_files = sorted(glob.glob(os.path.join(samples_dir, 'epoch_*_grid.png')))
        for g_file in grid_files:
            try:
                ep_num = int(os.path.basename(g_file).split('_')[1])
            except Exception:
                ep_num = 0
            wandb.log({
                'samples/grid': wandb.Image(g_file, caption=f"Epoch {ep_num}"),
                'epoch': ep_num
            })

    # 同步 loss_curve.png
    loss_curve_file = os.path.join(run_dir, 'loss_curve.png')
    if os.path.exists(loss_curve_file):
        wandb.log({'charts/loss_curve': wandb.Image(loss_curve_file)})

    run_url = run.url
    run.finish()
    print("=" * 60)
    print("同步完成！")
    print(f"可以在 W&B 上查看完整运行可视化: {run_url}")
    print("=" * 60)
    return run_url

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else None
    sync_run(target)

