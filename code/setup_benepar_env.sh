#!/bin/bash
# Setup script for Benepar parsing environment

echo "Creating Python virtual environment..."
python -m venv ~/benepar_env

echo "Activating environment..."
source ~/benepar_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch (CPU version for parsing)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing spaCy and language models..."
pip install spacy

# Download spaCy models for Chinese and French
python -m spacy download zh_core_web_sm
python -m spacy download fr_core_news_sm

echo "Installing benepar..."
pip install benepar

echo "Installing other dependencies..."
pip install tqdm pandas

echo "Downloading benepar models..."
python -c "
import benepar
# English model (for reference)
benepar.download('benepar_en3')
# Chinese model
benepar.download('benepar_zh2')
# We'll need a multilingual model for French
benepar.download('benepar_en3_large')
"

echo "Environment setup complete!"
echo "To activate: source ~/benepar_env/bin/activate"


