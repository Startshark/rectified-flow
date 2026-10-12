import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import sys
import json

if __name__ == '__main__':
    # 画Loss曲线看收敛情况
    target_path = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target_path = sys.argv[1]

    loss_list = None
    label_name = ''

    # 1. 如果用户给的是目录，优先检查该目录下的 loss_history.json 或 checkpoints
    if target_path and os.path.isdir(target_path):
        json_file = os.path.join(target_path, 'loss_history.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                loss_list = data.get('loss_list', [])
                label_name = os.path.basename(target_path)

    # 2. 如果尚未找到，寻找 .pth 权重文件
    if loss_list is None:
        ckpt_path = target_path if (target_path and target_path.endswith('.pth')) else None
        if not ckpt_path:
            # 搜索 outputs 和 checkpoints 目录下的最近 .pth 文件
            ckpts = []
            for search_dir in ['./outputs', './checkpoints']:
                if os.path.exists(search_dir):
                    for root, _, files in os.walk(search_dir):
                        for f in files:
                            if f.endswith('.pth'):
                                ckpts.append(os.path.join(root, f))
            if ckpts:
                ckpts.sort(key=os.path.getmtime)
                ckpt_path = ckpts[-1]

        if ckpt_path and os.path.exists(ckpt_path):
            print(f'Loading loss history from checkpoint: {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            loss_list = checkpoint.get('loss_list', [])
            label_name = os.path.basename(ckpt_path)

    if not loss_list:
        print('Error: 未找到有效的 loss 记录！请指定 checkpoints 文件或 outputs 目录。')
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Loss', alpha=0.5, color='dodgerblue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve ({label_name})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_file = 'loss_curve.png'
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Loss curve saved to {save_file}')
