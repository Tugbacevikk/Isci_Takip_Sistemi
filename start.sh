#!/bin/bash
# ============================================================
# İşçi Takip Sistemi - Linux / Raspberry Pi Başlatma Betiği
# ============================================================

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  İşçi Takip Sistemi - Raspberry Pi 5 Başlatılıyor"
echo "  Tarayıcıda açın: http://localhost:5000"
echo "============================================================"

# Sanal ortam kontrolü
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

python3 web/app.py
