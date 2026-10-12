#!/bin/bash
#SBATCH --job-name=rf-train
#SBATCH -A cis260266p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 1. 加载 Anaconda 模块并激活虚拟环境（静默处理非交互式 shell 的警告）
module load anaconda3/2024.10-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rectified-flow

# 2. 设置线程数、实时输出与工作目录
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
cd /ocean/projects/cis260266p/xzhai1/rectified-flow

# 3. 运行训练脚本（-u 实时刷新日志到 log 文件）
python -u train.py

echo "Job finished at $(date)"
