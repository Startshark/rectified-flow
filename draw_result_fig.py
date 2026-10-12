import matplotlib.pyplot as plt
import cv2
import os

# 读取results文件夹的100张图片
# img_folder = './results/reflow-cfg'  # v1.2
img_folder = 'results/cfg'  # v1.1
if not os.path.exists(img_folder):
    print(f"Error: 未找到图像目录 '{img_folder}'！")
    print("请先运行推理生成图片：python infer.py （或 sbatch infer.slurm）")
    exit(1)

img_files = [
    os.path.join(img_folder, f) for f in os.listdir(img_folder)
    if f.endswith('.png')
][:100]

if len(img_files) == 0:
    print(f"Error: '{img_folder}' 中没有找到 .png 格式的图像！")
    print("请先运行推理生成图片：python infer.py")
    exit(1)

# 按照自然数顺序排列
img_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))

for ax, img_file in zip(axes.flatten(), img_files):
    img = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
save_file = 'result_grid.png'
plt.savefig(save_file, bbox_inches='tight')
print(f'Result figure saved to {save_file}')
# plt.show()
