# ============================================================
# İş Takip Sistemi - Linux Docker Container Yapılandırması
# ============================================================

FROM python:3.11-slim-bookworm

# Linux Sistem Bağımlılıkları (OpenCV için gerekli shared kütüphaneler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma Dizinini Ayarla
WORKDIR /app

# Ortam Değişkenleri
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

# PyTorch CPU sürümünü (Hızlı 150MB indirme) önceden yükle
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

# Proje Dosyalarını Kopyala
COPY . .

# Web Portunu Dışarı Aç
EXPOSE 5000

# Uygulamayı Başlat
CMD ["python3", "web/app.py"]
