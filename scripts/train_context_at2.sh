MODEL_NAME=$1

GRES="gpu:a100:1"
EXCLUDE_ARGS="--exclude=deep-chungus-[1-5]"
PARTITION="high-priority"

output=$(COMMAND=generate MODEL_NAME=$MODEL_NAME sbatch --gres=$GRES $EXCLUDE_ARGS --partition=$PARTITION --job-name=generate --output=logs/generate_%a.log scripts/train_context_at2.sbatch)
generate_job_num=$(echo "$output" | awk '{print $NF}')

output=$(COMMAND=compute MODEL_NAME=$MODEL_NAME sbatch --gres=$GRES $EXCLUDE_ARGS --partition=$PARTITION --dependency $generate_job_num --job-name=compute --output=logs/compute_%a.log scripts/train_context_at2.sbatch)
compute_job_num=$(echo "$output" | awk '{print $NF}')

output=$(COMMAND=train MODEL_NAME=$MODEL_NAME sbatch --array=0 --gres=$GRES $EXCLUDE_ARGS --partition=$PARTITION --dependency $compute_job_num --job-name=train --output=logs/train.log scripts/train_context_at2.sbatch)
train_job_num=$(echo "$output" | awk '{print $NF}')
