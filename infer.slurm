#!/bin/bash
#SBATCH --job-name=rf-infer
#SBATCH -A cis260266p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH --output=infer_%j.log
#SBATCH --error=infer_%j.err

echo "Inference job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 1. 激活环境
module load anaconda3/2024.10-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rectified-flow

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
cd /ocean/projects/cis260266p/xzhai1/rectified-flow

# 2. 运行推理生成 100 张数字图像（保存至 results/cfg/）
echo "Generating images..."
python -u infer.py

# 3. 将 100 张图像拼接为 10x10 成果总览图 result_grid.png
echo "Drawing result grid..."
python -u draw_result_fig.py

echo "Inference and visualization finished at $(date)"

