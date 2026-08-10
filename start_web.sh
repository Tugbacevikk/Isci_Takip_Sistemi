#!/bin/bash
# ============================================================
# İşçi Takip Sistemi - Web Sunucusu Başlatma Betiği (Linux)
# ============================================================

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ -f "requirements_web.txt" ]; then
    pip3 install -r requirements_web.txt -q
fi

python3 web/app.py
