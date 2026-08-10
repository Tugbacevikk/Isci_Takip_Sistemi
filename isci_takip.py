"""
Fabrika İşçi Aktivite Takip Sistemi (ORM Yapısı)
- Linux ve Windows'ta çalışır (kamera backend'i otomatik seçilir)
- config.yaml'dan ayarları okur
- Sonuçları önce yerel SQLite'a (ORM), ardından merkezi PostgreSQL'e kaydeder

Kullanım: python isci_takip.py [config_yolu]
Çıkmak için: pencere açıkken 'q' tuşuna bas (headless modda Ctrl+C)
"""

import os
import sys
import time
import yaml
import cv2
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'core'))

from core.database.connection import DatabaseManager
from core.camera_manager import CameraProcessor
from pg_sync import SenkronThread

LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

file_handler = RotatingFileHandler(
    LOGS_DIR / 'isci_takip.log',
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler],
    force=True
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / '.env')
    except ImportError:
        pass
    cp = BASE_DIR / config_path
    if cp.exists():
        with open(cp, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

def parse_camera_id(cfg: dict):
    """
    Komut satırı argümanları, interaktif seçim veya config.yaml üzerinden kamera ID'sini belirler.
    Kullanım örnekleri:
      python isci_takip.py
      python isci_takip.py 1
      python isci_takip.py --select
      python isci_takip.py --camera 1
    """
    args = sys.argv[1:]
    
    # 1. Komut satırından "--select" veya "-s" veya "--sec" verilmişse interaktif tarat ve sor
    if "--select" in args or "-s" in args or "--sec" in args or cfg.get("ask_camera", False):
        print("\n[KAMERA TARAMA] Kullanılabilir kameralar taranıyor...")
        available = CameraProcessor.scan_cameras()
        if not available:
            print("Uyarı: Aktif kamera bulunamadı. Varsayılan (0) kullanılacak.")
            return 0
        
        first_id = available[0]['id'] if isinstance(available[0], dict) else available[0]
        print("\n--- Bulunan Kameralar ---")
        for item in available:
            if isinstance(item, dict):
                print(f"  [{item['id']}] {item['name']}")
            else:
                print(f"  [{item}] Kamera {item}")
        print("-------------------------")
        
        try:
            secim = input(f"Kullanmak istediğiniz kamera numarasını girin [Varsayılan: {first_id}]: ").strip()
            if secim and (secim.isdigit() or secim.startswith("rtsp://") or secim.startswith("http://")):
                return int(secim) if secim.isdigit() else secim
            return first_id
        except (KeyboardInterrupt, EOFError):
            print("\nVarsayılan kamera seçildi.")
            return first_id

    # 2. Komut satırından direk numara verilmişse (örn: python isci_takip.py 1 veya --camera 1)
    for i, arg in enumerate(args):
        if arg in ("--camera", "-c", "--kamera") and i + 1 < len(args):
            val = args[i + 1]
            return int(val) if val.isdigit() else val
        elif arg.isdigit():
            return int(arg)
        elif arg.startswith("rtsp://") or arg.startswith("http://"):
            return arg

    # 3. config.yaml içinden oku (Varsayılan 0)
    return cfg.get("camera_id", 0)

def main():
    print("Yapay zeka modelleri ve ORM veritabanı yükleniyor, lütfen bekleyin...")
    cfg = load_config()
    cam_id = parse_camera_id(cfg)
    cam_devices = CameraProcessor.get_camera_device_names()
    cam_name = cam_devices[cam_id] if (isinstance(cam_id, int) and cam_id < len(cam_devices)) else f"Kamera {cam_id}"
    logger.info(f"Seçilen kamera: {cam_name} (ID: {cam_id})")

    db_path = BASE_DIR / cfg.get('db_path', 'isci_takip.db')
    db_mgr = DatabaseManager(str(db_path))

    processor = CameraProcessor(camera_id=cam_id, config=cfg, db_path=str(db_path))
    if not processor.start_camera():
        print("Hata: Kamera açılamadı.")
        return

    # PG Senkronizasyon thread (Koşulsuz Otomatik Senkronizasyon)
    merkezi_db_cfg = cfg.get("merkezi_db", {}) or cfg
    senkron_thread = None
    try:
        senkron_thread = SenkronThread(
            db_mgr=db_mgr,
            merkezi_db_cfg=merkezi_db_cfg,
            istasyon_adi=cfg.get("istasyon_adi", "auto")
        )
        senkron_thread.start()
        logger.info("Otomatik PostgreSQL senkronizasyon thread'i başlatıldı.")
    except Exception as e:
        logger.error(f"PG senkronizasyon başlatılamadı: {e}")

    # ADIM 2: Headless (Ekransız) Mod & DISPLAY Kontrolü
    headless_cfg = cfg.get('headless_mode', False)
    display_available = sys.platform == 'win32' or bool(os.environ.get('DISPLAY'))
    
    goster = cfg.get('goster', True) and display_available and not headless_cfg

    if not display_available:
        logger.info("Ekransız (Linux/Headless) ortam tespit edildi. cv2.imshow penceresi devre dışı bırakıldı.")
    elif headless_cfg:
        logger.info("Headless mod aktif. Görüntü penceresi açılmayacak.")

    print("Hibrit Takip Sistemi Başlatıldı. Çıkmak için 'q' tuşuna basın (Headless modda Ctrl+C).")
    
    try:
        while processor.is_running:
            frame = processor.get_current_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            if goster:
                try:
                    cv2.imshow("Fabrika Isci Takip Sistemi", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Kullanıcı tarafından durduruldu.")
                        break
                except Exception as e:
                    logger.warning(f"cv2.imshow çağrısı çökme koruması devreye girdi ({e}). Headless moda geçiliyor.")
                    goster = False
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("Sistem kapatıldı.")
    finally:
        processor.stop_camera()
        if goster:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if senkron_thread:
            print("[PG] Senkronizasyon thread'i durduruluyor...")
            senkron_thread.durdur()

if __name__ == '__main__':
    main()
