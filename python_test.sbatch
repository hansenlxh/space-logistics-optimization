#!/bin/bash
#SBATCH -python_test                # job name
#SBATCH -A gts-kho34                     # account to which job is charged, gts-kho34-fy20phase1 if no balance on gts-kho34
#SBATCH -N 1	 # number of nodes equired
#SBATCH -n 24	 # number of cores per node required
#SBATCH --mem-per-cpu=2G                 # memory per core
#SBATCH -t 30                             # duration of the job (e.g. 30 min)
#SBATCH -o Report-%j.out                  # output file name
#SBATCH --mail-type=BEGIN,END,FAIL              # E-Mail preferences
#SBATCH --mail-user=misaji3@gatech.edu        # E-mail address for notifications

cd $SLURM_SUBMIT_DIR  # dir from which this .sbatch file is submitted
eval "$(conda shell.bash hook)"
conda activate slpy
pytest
