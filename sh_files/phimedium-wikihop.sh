#!/bin/bash
#SBATCH --no-requeue
#SBATCH --account=dssc
#SBATCH --job-name="phiWikiM"
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1                  # <-- TODO adjust this
#SBATCH --mem=20G                             # <-- TODO adjust this
#SBATCH --time=04:00:00                       # <-- TODO adjust this
#SBATCH --output=slurm_outs/base-phimedium-wikihop.out


# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"


# Needed sourcing
source /u/dssc/scandu00/nlp-env/bin/activate

# Needed modules
# module load <module_name>

# Needed exports
# export <export_name>=<export_value>
#variables
FILE_NAME=/u/dssc/scandu00/tesi-bozze/def/phi-medium/baseline/baseline-phimedium-wikihop.py

CMD="python3 -u"

if [ ! -f "$FILE_NAME" ]; then
  echo "The file $FILE_NAME does not exist"
  exit 1
fi

# Other checks there

$CMD $FILE_NAME

echo "DONE!"
