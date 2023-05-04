#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=taichi-nerfs
#SBATCH --mail-type=END
#SBATCH --mail-user=yl5680@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge

singularity exec --nv \
            --overlay /scratch/yl5680/taichi_nerf.ext3:rw \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate taichi_nerf; cd /scratch/yl5680/taichi-nerfs; scripts/train_nsvf_lego.sh"


