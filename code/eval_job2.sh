#!/bin/bash --login

#SBATCH -p multicore

#SBATCH -n 12

##SBATCH -t 2-0

#SBATCH -t 06:00:00

# Mail events: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.cui-3@postgrad.manchester.ac.uk

# Job name
#SBATCH -J senteval_translation

module purge
module load compilers/gcc/9.3.0
module load apps/binapps/anaconda3/2022.10
#module load libs/cuda/12.2.2
#module load libs/cuda/11.7.0
# module load libs/cuda/8.0.61


#after this line
##------------------------env setting--------------------##
#conda env create -f unveiling-bf16.yaml -n brain_encoding2
conda info -e
#conda create -n brain_encoding
#nvcc --version
#nvidia-smi
#source activate brain_encoding
source activate Alice_encoding
#pip install concurrent
#pip install brainscore
#pip install brainio
#pip install sklearn
#pip install tqdm
#pip install scipy
#pip install rsatoolbox
#pip install --upgrade huggingface_hub

##cuda not available的解决
#conda uninstall pytorch torchvision torchaudio -y
#pip uninstall torch torchvision torchaudio -y
#pip uninstall numpy -y
#pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 # Using CUDA 11.8 as it's more readily available with recent PyTorch versions via pip and often compatible with 11.7 drivers.
#pip install --upgrade "numpy<2.0" scipy scikit-learn pandas psutil transformers regex sentencepiece # Pinned numpy version


# pip install google-genai

# export GOOGLE_API_KEY=""

# python SentEval/code/translate_coordination_gemini.py \
#     --input_file SentEval/data/probing/coordination_inversion.txt \
#     --output_dir SentEval/data_fr/probing \
#     --output_name coordination_inversion.txt \
#     --batch_size 16 \
#     --max_workers 12 \
#     --target_lang fr

source deactivate

# Run the Nvidia app that reports GPU statistics
