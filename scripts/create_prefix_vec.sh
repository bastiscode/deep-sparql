#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name=prefix_vec
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --time=4:00:00

script_dir=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_dir)
text_utils=$script_dir/../third_party/text-utils

index=${INDEX?"env var INDEX not found"}
out=${OUT?"env var OUT not found"}
cmd="$text_utils/scripts/create_prefix_vec.py --file $index --out $out"

tokenizer_cfg=${TOKENIZER_CFG:-""}
if [[ $tokenizer_cfg != "" ]]; then
    cmd="$cmd --tokenizer-cfg $tokenizer_cfg"
fi

python $cmd
