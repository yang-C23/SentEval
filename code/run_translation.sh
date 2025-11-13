#!/bin/bash --login

#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -c 16

#SBATCH -t 2-0

# Mail events: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.cui-3@postgrad.manchester.ac.uk

# Job name
#SBATCH -J senteval_translation

echo "========================================"
echo "Starting SentEval Translation Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of GPUs: $SLURM_GPUS"
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "========================================"

# Load required modules
module purge
module load compilers/gcc/9.3.0
module load apps/binapps/anaconda3/2022.10
module load libs/cuda/8.0.61

# Set working directory
cd /mnt/iusers01/fatpou01/compsci01/v07051yc/SentEval

# Check if conda environment exists, if not create it
ENV_NAME="senteval_translation"
echo "Checking for conda environment: $ENV_NAME"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.9 -y
    
    source activate $ENV_NAME
    
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers>=4.30.0
    pip install sentencepiece
    pip install protobuf
    pip install accelerate
    pip install jieba
    pip install sacremoses
    pip install tqdm
    
    echo "Environment setup complete"
else
    echo "Environment $ENV_NAME already exists"
    source activate $ENV_NAME
fi

# Verify installation
echo "========================================"
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "Number of GPUs:"
python -c "import torch; print(torch.cuda.device_count())"
echo "========================================"



# Define file paths
INPUT_FILE="data/probing/sentence_length.txt"
TOTAL_LINES=119988
HALF_LINES=59994

# Create temporary directory for partial translations
mkdir -p code/tmp

echo "========================================"
echo "Starting translation process"
echo "Total sentences to translate: $TOTAL_LINES"
echo "Using 2 GPUs for parallel processing"
echo "========================================"

# Function to translate for a specific language
translate_language() {
    local LANG=$1
    local LANG_NAME=$2
    
    echo "========================================"
    echo "Translating to $LANG_NAME ($LANG)"
    echo "========================================"
    
    # Define output paths
    if [ "$LANG" == "zh" ]; then
        OUTPUT_DIR="data_zh/probing"
    else
        OUTPUT_DIR="data_fr/probing"
    fi
    
    mkdir -p $OUTPUT_DIR
    
    # Translation phase - parallel processing with 2 GPUs
    echo "Phase 1: Translation with parallel GPUs"
    echo "GPU 0: Processing lines 0-$HALF_LINES"
    echo "GPU 1: Processing lines $HALF_LINES-$TOTAL_LINES"
    
    # Start translation on GPU 0 (first half)
    python code/translate_probing.py \
        --input_file $INPUT_FILE \
        --output_file code/tmp/translated_${LANG}_part1.txt \
        --target_lang $LANG \
        --gpu_id 0 \
        --start_idx 0 \
        --end_idx $HALF_LINES \
        --batch_size 8 \
        > code/tmp/translate_${LANG}_gpu0.log 2>&1 &
    PID1=$!
    
    # Start translation on GPU 1 (second half)
    python code/translate_probing.py \
        --input_file $INPUT_FILE \
        --output_file code/tmp/translated_${LANG}_part2.txt \
        --target_lang $LANG \
        --gpu_id 1 \
        --start_idx $HALF_LINES \
        --end_idx $TOTAL_LINES \
        --batch_size 8 \
        > code/tmp/translate_${LANG}_gpu1.log 2>&1 &
    PID2=$!
    
    echo "Waiting for translation to complete..."
    echo "GPU 0 PID: $PID1"
    echo "GPU 1 PID: $PID2"
    
    # Wait for both processes to complete
    wait $PID1
    STATUS1=$?
    echo "GPU 0 completed with status: $STATUS1"
    
    wait $PID2
    STATUS2=$?
    echo "GPU 1 completed with status: $STATUS2"
    
    # Check if both completed successfully
    if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
        echo "ERROR: Translation failed!"
        echo "GPU 0 status: $STATUS1"
        echo "GPU 1 status: $STATUS2"
        echo "Check log files:"
        echo "  code/tmp/translate_${LANG}_gpu0.log"
        echo "  code/tmp/translate_${LANG}_gpu1.log"
        return 1
    fi
    
    # Merge translated files
    echo "Phase 2: Merging translated files"
    cat code/tmp/translated_${LANG}_part1.txt code/tmp/translated_${LANG}_part2.txt > code/tmp/translated_${LANG}_merged.txt
    
    # Count lines
    MERGED_LINES=$(wc -l < code/tmp/translated_${LANG}_merged.txt)
    echo "Merged file has $MERGED_LINES lines (expected: $TOTAL_LINES)"
    
    if [ $MERGED_LINES -ne $TOTAL_LINES ]; then
        echo "WARNING: Line count mismatch!"
    fi
    
    # Re-labeling phase
    echo "Phase 3: Re-labeling based on word count"
    python code/relabel.py \
        --input_file code/tmp/translated_${LANG}_merged.txt \
        --output_file ${OUTPUT_DIR}/sentence_length.txt \
        --language $LANG
    
    if [ $? -eq 0 ]; then
        echo "Successfully created ${OUTPUT_DIR}/sentence_length.txt"
        
        # Show statistics
        echo "Final file statistics:"
        wc -l ${OUTPUT_DIR}/sentence_length.txt
        echo "Label distribution:"
        cut -f2 ${OUTPUT_DIR}/sentence_length.txt | sort | uniq -c
        echo "Partition distribution:"
        cut -f1 ${OUTPUT_DIR}/sentence_length.txt | sort | uniq -c
    else
        echo "ERROR: Re-labeling failed!"
        return 1
    fi
    
    echo "========================================"
    echo "Completed translation to $LANG_NAME"
    echo "========================================"
}

# Translate to Chinese
translate_language "zh" "Chinese"
ZH_STATUS=$?

# Translate to French
translate_language "fr" "French"
FR_STATUS=$?

# Summary
echo "========================================"
echo "Translation Job Summary"
echo "========================================"
echo "Chinese translation status: $ZH_STATUS"
echo "French translation status: $FR_STATUS"

if [ $ZH_STATUS -eq 0 ] && [ $FR_STATUS -eq 0 ]; then
    echo "SUCCESS: All translations completed successfully"
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -rf code/tmp
    
    echo "Final output files:"
    ls -lh data_zh/probing/sentence_length.txt
    ls -lh data_fr/probing/sentence_length.txt
    
    echo ""
    echo "You can now validate the outputs:"
    echo "  python code/validate_output.py --input_file data_zh/probing/sentence_length.txt"
    echo "  python code/validate_output.py --input_file data_fr/probing/sentence_length.txt"
    
    exit 0
else
    echo "FAILURE: Some translations failed"
    echo "Temporary files kept for debugging in code/tmp/"
    exit 1
fi

