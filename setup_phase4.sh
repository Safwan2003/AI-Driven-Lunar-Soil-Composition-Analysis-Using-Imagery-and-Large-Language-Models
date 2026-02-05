#!/bin/bash
# Complete Phase 4 Setup Script
# Run this to set up the complete lunar analysis system

set -e  # Exit on any error

echo "🌙 SUPARCO Lunar Analysis System - Phase 4 Setup"
echo "=" | head -c 60 | tr '\n' '='
echo ""

# 1. Create directories
echo "📁 Creating directory structure..."
mkdir -p src/models_data
mkdir -p labeled_data/terrain/{rocky_region,crater,big_rock,artifact}
mkdir -p labeled_data/composition
mkdir -p labeled_data/crops

# 2. Check if SAM 2.1 assets exist
echo ""
echo "🔍 Checking SAM 2.1 assets..."

if [ ! -f "src/models_data/sam2.1_hiera_t.yaml" ]; then
    echo "⬇️  Downloading SAM 2.1 config..."
    curl -L https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
        -o src/models_data/sam2.1_hiera_t.yaml
fi

if [ ! -f "src/models_data/sam2.1_hiera_tiny.pt" ]; then
    echo "⬇️  Downloading SAM 2.1 weights (38MB, this may take a minute)..."
    curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
        -o src/models_data/sam2.1_hiera_tiny.pt
fi

echo "✅ SAM 2.1 assets ready"

# 3. Set Python path
export PYTHONPATH=$PYTHONPATH:.

# 4. Test imports
echo ""
echo "🧪 Testing imports..."
python -c "from src.composition import LuceyHeuristicEstimator; print('✅ Composition module OK')"
python -c "from src.analysis import LunarAnalysisPipeline; print('✅ Pipeline module OK')"

# 5. Launch app
echo ""
echo "🚀 Launching Streamlit app..."
echo "Open browser to: http://localhost:8501"
echo ""
streamlit run src/ui/app.py
