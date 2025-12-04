#!/bin/bash --login

#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -c 8

#SBATCH -t 2-0

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.cui-3@postgrad.manchester.ac.uk

#SBATCH -J relabel_constituents

module purge
module load compilers/gcc/9.3.0
module load apps/binapps/anaconda3/2022.10
module load libs/cuda/11.7.0

echo "=========================================="
echo "Top Constituents Relabeling Pipeline"
echo "=========================================="
echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES"

# Recreate environment with GPU support
echo "Setting up benepar environment with GPU support..."

# Remove old environment and create new one
rm -rf $HOME/benepar_env
python -m venv $HOME/benepar_env
source $HOME/benepar_env/bin/activate

# Install GPU-compatible torch first
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies with compatible versions
pip install transformers==4.30.0 spacy benepar tqdm pandas

# Download spaCy and benepar models
python -m spacy download zh_core_web_sm
python -m spacy download fr_core_news_sm
python -c "import benepar; benepar.download('benepar_zh2')"

echo ""
echo "Python: $(which python)"
echo "Torch version and CUDA:"
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test on small sample first
echo ""
echo "=========================================="
echo "Testing on small sample (500 sentences)..."
echo "=========================================="

# Chinese test
echo ""
echo "Testing Chinese..."
python SentEval/code/relabel_top_constituents.py \
    --input_file SentEval/data_zh/probing/top_constituents.txt \
    --output_file SentEval/data_zh/probing/top_constituents_test.txt \
    --language zh \
    --sample_count 500 \
    --verbose

# Check if test successful
if [ $? -eq 0 ]; then
    echo "Test successful! Proceeding with full dataset..."
else
    echo "Test failed! Exiting."
    exit 1
fi

# Full Chinese processing
echo ""
echo "=========================================="
echo "Processing full Chinese dataset..."
echo "=========================================="

python SentEval/code/relabel_top_constituents.py \
    --input_file SentEval/data_zh/probing/top_constituents.txt \
    --output_file SentEval/data_zh/probing/top_constituents_relabeled.txt \
    --language zh \
    --target_classes 20 \
    --verbose

# French processing
echo ""
echo "=========================================="
echo "Processing full French dataset..."
echo "=========================================="

python SentEval/code/relabel_top_constituents.py \
    --input_file SentEval/data_fr/probing/top_constituents.txt \
    --output_file SentEval/data_fr/probing/top_constituents_relabeled.txt \
    --language fr \
    --target_classes 20 \
    --verbose

echo ""
echo "=========================================="
echo "Verifying output files..."
echo "=========================================="

echo "Chinese output:"
wc -l SentEval/data_zh/probing/top_constituents_relabeled.txt
cut -f2 SentEval/data_zh/probing/top_constituents_relabeled.txt | sort | uniq -c | sort -rn | head -25

echo ""
echo "French output:"
wc -l SentEval/data_fr/probing/top_constituents_relabeled.txt
cut -f2 SentEval/data_fr/probing/top_constituents_relabeled.txt | sort | uniq -c | sort -rn | head -25

echo ""
echo "=========================================="
echo "Relabeling Complete!"
echo "=========================================="

deactivate

