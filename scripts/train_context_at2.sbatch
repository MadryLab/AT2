#!/bin/bash
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --requeue

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m scripts.train_context_at2 --command $COMMAND --model_name $MODEL_NAME --job_index $SLURM_ARRAY_TASK_ID --num_jobs 10
