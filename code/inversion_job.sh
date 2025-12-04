#!/bin/bash --login

#SBATCH -p multicore

#SBATCH -n 4

#SBATCH -t 01:00:00

# Mail events: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.cui-3@postgrad.manchester.ac.uk

# Job name
#SBATCH -J coordination_inversion

module purge
module load compilers/gcc/9.3.0
module load apps/binapps/anaconda3/2022.10

conda info -e
source activate Alice_encoding

echo "=========================================="
echo "Starting Coordination Inversion Pipeline"
echo "=========================================="

# Process Chinese data
echo ""
echo "Processing Chinese data..."
python SentEval/code/apply_coordination_inversion.py \
    --input_file SentEval/data_zh/probing/coordination_inversion.txt \
    --output_file SentEval/data_zh/probing/coordination_inversion_final.txt \
    --language zh \
    --inversion_ratio 0.5 \
    --seed 42 \
    --remove_english \
    --verbose

# Process French data
echo ""
echo "Processing French data..."
python SentEval/code/apply_coordination_inversion.py \
    --input_file SentEval/data_fr/probing/coordination_inversion.txt \
    --output_file SentEval/data_fr/probing/coordination_inversion_final.txt \
    --language fr \
    --inversion_ratio 0.5 \
    --seed 42 \
    --remove_english \
    --verbose

echo ""
echo "=========================================="
echo "Coordination Inversion Complete!"
echo "=========================================="

source deactivate

