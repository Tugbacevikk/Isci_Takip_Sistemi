# 🏭 Fabrika İş ve Aktivite Takip Sistemi

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8%20%7C%20YuNet%20%7C%20SFace-00FFFF?style=for-the-badge)
![Flask](https://img.shields.io/badge/Web-Flask%20%7C%20SocketIO-000000?style=for-the-badge&logo=flask)
![Database](https://img.shields.io/badge/Database-SQLite%20%2B%20PostgreSQL-336791?style=for-the-badge&logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=for-the-badge&logo=docker)

**Fabrika İş ve Aktivite Takip Sistemi**, üretim tesislerinde ve istasyonlarda yapay zeka (Bilgisayarlı Görü) destekli canlı iş takibi, yüz tanıma, duruş/pozisyon analizi (Pose Estimation), özel kaynak (welding) işlemi tespiti ve bölge (ROI) bazlı hareket/hareketsizlik analizi gerçekleştiren kapsamlı bir endüstriyel takip sistemidir.

---

## 🚀 Öne Çıkan Özellikler

- 🎯 **Gelişmiş Yapay Zeka & Bilgisayarlı Görü (AI & Vision)**:
  - **YOLOv8 Pose Estimation**: İşçi duruş, pozisyon ve iskelet takibi.
  - **YuNet & SFace Integration**: Yüksek doğrulukta yüz tespiti ve kayıtlı işçi veri tabanı ile anlık yüz tanıma.
  - **Özel Eğitilmiş Kaynak (Welding) Modeli**: Kaynak makineleri ve iş istasyonlarındaki kaynak kıvılcımı/aktivitesini anlık algılama.
- 📹 **Kamera & İstasyon Yönetimi**:
  - WebCam, IP Kamera (RTSP/HTTP akışları) ve video dosyaları desteği.
  - İnteraktif kamera tarama (`--select`) ve otomatik cihaz bağlama.
  - İstasyon ve Bölge (ROI - Region of Interest) sınırları belirleme, alan ihlali ve hareket/hareketsizlik takibi.
- 🌐 **Modern Web Kontrol Paneli**:
  - Flask + SocketIO ile düşük gecikmeli canlı video akışı.
  - Canlı istasyon takibi, performans istatistikleri ve alarm paneli.
  - İşçi yönetimi (Fotoğraf yükleme, yüz embedding oluşturma ve profil yönetimi).
  - Tarih aralıklı raporlama ve Excel/CSV export imkanı.
- 🔄 **Çift Veritabanı & Hibrit Senkronizasyon (Edge Computing)**:
  - İnternet kopmalarına dirençli yerel **SQLite** kaydı (Edge/Local).
  - Arka planda çalışan senkronizasyon thread'i ile merkezi **PostgreSQL** veritabanına otomatik veri aktarımı.
- 🐳 **Endüstriyel Dağıtım Desteği**:
  - Docker & Docker Compose desteği.
  - Linux `systemd` servis entegrasyonu ve Windows `.bat` başlatıcıları.

---

## 📁 Proje Dizin Yapısı

```bash
istakip/
├── 📄 isci_takip.py           # Ana AI İşçi ve Aktivite Takip Servisi
├── 📁 web/                     # Web Arayüzü (Flask Uygulaması)
│   ├── app.py                  # Flask REST API & WebSocket Sunucusu
│   ├── 📁 templates/           # Jinja2 HTML Şablonları (Dashboard, İşçiler, Alarmlar vb.)
│   └── 📁 static/              # CSS, JavaScript ve İşçi / Video Yüklemeleri
├── 📁 core/                    # Çekirdek Modüller
│   ├── camera_manager.py       # Kamera Okuma, Görüntü İşleme ve AI Modeli Yürütücü
│   └── 📁 database/            # SQLAlchemy ORM Bağlantıları ve Tablo Modelleri
├── 📁 models/                  # AI Model Ağırlıkları
│   ├── yolov8n.pt              # Nesne Algılama
│   ├── yolov8n-pose.pt         # Pozisyon / İskelet Takibi
│   ├── welding_det.pt          # Özel Eğitilmiş Kaynak Algılama Modeli
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── 📄 config.yaml              # Kamera, ROI ve Sistem Yapılandırma Dosyası
├── 📄 pg_kurulum.py            # PostgreSQL Veritabanı Tablo Oluşturucu
├── 📄 pg_sync.py               # SQLite -> PostgreSQL Arka Plan Senkronizatörü
├── 📄 requirements.txt         # Takip Servisi Bağımlılıkları
├── 📄 requirements_web.txt     # Web Arayüzü Bağımlılıkları
├── 📄 docker-compose.yml       # Docker Konfigürasyonu
└── 📄 Dockerfile               # Docker İmaj Yapılandırması
```

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimler

- **Python**: 3.10 veya üzeri
- **PostgreSQL**: (İsteğe bağlı, merkezi veritabanı senkronizasyonu için)
- **Git**

### 2. Depoyu Klonlayın ve Bağımlılıkları Yükleyin

```bash
git clone https://github.com/Tugbacevikk/Is_Takip_Sistemi.git
cd Is_Takip_Sistemi

# Sanal ortam oluşturun (Önerilen)
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/macOS için:
source venv/bin/activate

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
pip install -r requirements_web.txt
```

### 3. Ortam Değişkenleri ve Konfigürasyon

Proje kök dizinindeki `.env.example` dosyasını `.env` olarak kopyalayın ve PostgreSQL / Sistem bilgilerinizi düzenleyin:

```bash
cp .env.example .env
```

`config.yaml` dosyasından kamera, ROI oranları, alarm eşik değerleri ve merkezi veritabanı ayarlarını ihtiyacınıza göre özelleştirebilirsiniz.

### 4. Veritabanını Hazırlayın

Merkezi PostgreSQL veritabanı tablolarını oluşturmak için:

```bash
python pg_kurulum.py
```

### 5. Uygulamayı Başlatın

#### A. AI Takip Servisini Başlatma (Görüntü İşleme)

```bash
# Varsayılan kamera ile başlatma:
python isci_takip.py

# İnteraktif kamera seçimi ile başlatma:
python isci_takip.py --select

# Belirli bir kamera ID veya RTSP adresi ile:
python isci_takip.py 0
```

#### B. Web Paneli Başlatma

```bash
python web/app.py
```
Web arayüzüne tarayıcınızdan **`http://localhost:5000`** (veya konfigüre edilen port) üzerinden erişebilirsiniz.

Windows üzerinde hem Web Paneli hem Takip Servisini tek tıkla başlatmak için:
```cmd
start_web.bat
```

---

## 🐳 Docker ile Çalıştırma

Projeyi konteynerize edilmiş ortamda çalıştırmak için Docker Compose kullanabilirsiniz:

```bash
# Konteynerleri derleyin ve arka planda çalıştırın
docker-compose up -d --build

# Logları takip etmek için
docker-compose logs -f
```

---

## ⚙️ Yapılandırma (`config.yaml`)

| Parametre | Açıklama | Varsayılan |
| :--- | :--- | :--- |
| `camera_id` | Kamera cihaz indeksi (0, 1) veya RTSP URL | `0` |
| `station_name` | İstasyon / Bölge adı | `Istasyon-2` |
| `welding_conf` | Kaynak tespiti minimum güven eşiği | `0.3` |
| `hareket_esik_orani` | Hareket algılama duyarlılık oranı | `0.04` |
| `headless_mode` | Ekran çıktısı olmadan çalışma modu | `true` |
| `merkezi_db.aktif` | PostgreSQL senkronizasyonunun aktifliği | `true` |
| `merkezi_db.senkron_araligi_sn` | Senkronizasyon periyodu (saniye) | `60` |

---

## 🤝 Katkıda Bulunma

1. Bu depoyu çatallayın (Fork).
2. Yeni bir özellik dalı oluşturun (`git checkout -b feature/YeniOzellik`).
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`).
4. Dalınıza push yapın (`git push origin feature/YeniOzellik`).
5. Bir Pull Request (PR) oluşturun.

---

## 🔒 Telif Hakkı & Kullanım

Bu proje özel bir yazılımdır. Tüm hakları saklıdır.

