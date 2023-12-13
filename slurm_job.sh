#!/bin/bash
#SBATCH -J arcInference                    # Job name
#SBATCH -n1                                     # Number of cores required
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o generateOutputs-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=bgoyal7@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

source ~/.bashrc
# module load anaconda3/2022.05.0.1                   # Load module dependencies
conda activate vitinf
srun python generate_outputs.py