import matplotlib.pyplot as plt
import torch
import os
import sys

if __name__ == '__main__':
    # 画Loss曲线看收敛情况
    ckpt_path = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        ckpt_path = sys.argv[1]
    elif os.path.exists('./checkpoints/v1.1-cfg/miniunet_49.pth'):
        ckpt_path = './checkpoints/v1.1-cfg/miniunet_49.pth'
    elif os.path.exists('./checkpoints/v1.2-reflow-cfg/miniunet_19.pth'):
        ckpt_path = './checkpoints/v1.2-reflow-cfg/miniunet_19.pth'
    else:
        ckpts = []
        for root, _, files in os.walk('./checkpoints'):
            for f in files:
                if f.endswith('.pth'):
                    ckpts.append(os.path.join(root, f))
        if ckpts:
            ckpts.sort(key=os.path.getmtime)
            ckpt_path = ckpts[-1]

    if not ckpt_path or not os.path.exists(ckpt_path):
        print('Error: 未找到权重文件！请检查 checkpoints 目录。')
        sys.exit(1)

    print(f'Loading loss history from: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    loss_list = checkpoint.get('loss_list', [])

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve ({os.path.basename(ckpt_path)})')
    plt.tight_layout()
    plt.savefig('loss_curve.png', bbox_inches='tight')
    print('Loss curve saved to loss_curve.png')
