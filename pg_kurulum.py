"""
pg_kurulum.py - Merkezi PostgreSQL Tablo Kurulum Scripti (Code-First)
"""
import yaml
import logging
from pathlib import Path
from pg_sync import pg_baglan, pg_tablo_hazirla, pg_baglantiyi_kapat

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        logger.error("config.yaml bulunamadı.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    merkezi_cfg = cfg.get("merkezi_db", {})
    logger.info("PostgreSQL veritabanına bağlanılıyor...")
    engine = pg_baglan(merkezi_cfg)
    if engine:
        logger.info("Code-First ORM ile tablolar ve indeksler hazırlanıyor...")
        if pg_tablo_hazirla(engine):
            logger.info("PostgreSQL Code-First kurulumu başarıyla tamamlandı.")
        else:
            logger.error("PostgreSQL tablo kurulumu başarısız oldu.")
        pg_baglantiyi_kapat(engine)
    else:
        logger.error("PostgreSQL veritabanına bağlanılamadı.")


if __name__ == '__main__':
    main()
