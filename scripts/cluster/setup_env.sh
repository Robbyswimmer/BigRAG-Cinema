#!/usr/bin/env bash
# One-time environment setup for BigRAG Cinema on a SLURM cluster.
# Run interactively on a login node:
#   bash scripts/cluster/setup_env.sh

set -euo pipefail

ENV_NAME="bigrag"
PYTHON_VERSION="3.9"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== BigRAG Cinema — Environment Setup ==="
echo "Project directory: $PROJECT_DIR"

# ── Create conda environment ──────────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists — updating."
else
    echo "Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION) ..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Activate (works inside a script with conda init)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Install Java 11 (needed by Spark) ────────────────────────────────
echo "Installing OpenJDK 11 via conda ..."
conda install -y -c conda-forge openjdk=11

# ── Install Python dependencies ──────────────────────────────────────
echo "Installing Python requirements ..."
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# ── Install project in editable mode ─────────────────────────────────
echo "Installing bigrag-cinema in editable mode ..."
pip install -e "$PROJECT_DIR"

# ── Verify ────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
echo "Python : $(python --version)"
echo "Java   : $(java -version 2>&1 | head -1)"
echo "PySpark: $(python -c 'import pyspark; print(pyspark.__version__)')"
echo "bigrag : $(python -c 'import bigrag; print(bigrag.__file__)')"
echo ""
echo "Setup complete. Activate with:  conda activate $ENV_NAME"
