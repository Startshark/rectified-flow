#!/bin/bash
# 申请 PSC Bridges-2 GPU-shared 分区的交互式节点
# 默认申请 1 小时，1 块 V100-32GB GPU，5 个 CPU 核心，16GB 内存
TIME=${1:-01:00:00}

echo "正在申请交互式 GPU 节点 (时长: $TIME)..."
srun -A cis260266p -p GPU-shared --gres=gpu:v100-32:1 --cpus-per-task=5 --mem=16G -t $TIME --pty bash

