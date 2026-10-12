import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import sys

# 优先从命令行参数读取图像文件夹，否则查找最新 outputs 或 results
img_folder = None
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img_folder = sys.argv[1]
else:
    # 查找 outputs 下最新的 samples 子目录
    candidate_folders = []
    if os.path.exists('./outputs'):
        for root, dirs, files in os.walk('./outputs'):
            if any(f.endswith('.png') for f in files) and 'samples' in root:
                candidate_folders.append(root)
    if candidate_folders:
        candidate_folders.sort(key=os.path.getmtime)
        img_folder = candidate_folders[-1]
    elif os.path.exists('results/cfg'):
        img_folder = 'results/cfg'
    elif os.path.exists('results/reflow-cfg'):
        img_folder = 'results/reflow-cfg'

if not img_folder or not os.path.exists(img_folder):
    print(f"Error: 未找到图像目录！")
    print("使用方法: python draw_result_fig.py [图片文件夹路径]")
    sys.exit(1)

print(f"Loading images from: {img_folder}")
img_files = [
    os.path.join(img_folder, f) for f in os.listdir(img_folder)
    if f.endswith('.png') and not f.endswith('_grid.png')
]

if len(img_files) == 0:
    print(f"Error: '{img_folder}' 中没有找到单张 .png 格式的图像！")
    sys.exit(1)

# 尝试按照数字序号排序
try:
    img_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0].split('.')[0]))
except Exception:
    img_files.sort()

# 最多显示100张
img_files = img_files[:100]
n = len(img_files)
cols = 10 if n >= 10 else n
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))
axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for i, ax in enumerate(axes_flat):
    if i < len(img_files):
        img = cv2.imread(img_files[i])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
    ax.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()
save_file = 'result_grid.png'
plt.savefig(save_file, bbox_inches='tight', dpi=150)
plt.close()
print(f'Result figure saved to {save_file}')
