#!/bin/bash
echo "Setting up Insurance RAG System"

# Create directories
mkdir -p src tests evaluation scripts documents logs cache chroma_db

# Create virtual environment
python3 -m venv venv
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
"

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env .env.backup 2>/dev/null || true
    cat > .env << 'EOF'
GEMINI_API_KEY=your_gemini_api_key_here
DOCUMENTS_DIR=./documents
LOGS_DIR=./logs
CACHE_DIR=./cache
CHROMA_DB_PATH=./chroma_db
TOP_K_RESULTS=8
SIMILARITY_THRESHOLD=0.5
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=100
LOG_LEVEL=INFO
EOF
fi

echo "Setup complete. Edit .env to add your GEMINI_API_KEY."