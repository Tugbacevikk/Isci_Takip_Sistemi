import os
import sys
import logging
from sqlalchemy import text
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, '.env'))

from core.database.connection import DatabaseManager
from core.database.models import Base, User, Worker
from pg_sync import pg_baglan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    logger.info("Veritabanı Patron/Firma migrasyonu başlatılıyor...")
    
    # 1. SQLite Migrasyonu
    db_path = os.path.join(BASE_DIR, 'isci_takip.db')
    db_mgr = DatabaseManager(db_path)
    
    with db_mgr.engine.connect() as conn:
        # SQLite columns check & add
        try:
            conn.execute(text("ALTER TABLE workers ADD COLUMN patron_id INTEGER REFERENCES users(id) ON DELETE SET NULL"))
            conn.commit()
            logger.info("SQLite 'workers' tablosuna 'patron_id' sütunu eklendi.")
        except Exception as e:
            logger.info(f"SQLite patron_id var veya eklendi ({e})")
            
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN firma_adi VARCHAR(150)"))
            conn.commit()
            logger.info("SQLite 'users' tablosuna 'firma_adi' sütunu eklendi.")
        except Exception as e:
            logger.info(f"SQLite firma_adi var veya eklendi ({e})")

    # Ensure default Super Admin / Main Patron User exists in SQLite
    with db_mgr.get_session() as session:
        admin = session.query(User).filter(User.kullanici_adi == 'admin').first()
        if not admin:
            from werkzeug.security import generate_password_hash
            admin = User(
                kullanici_adi='admin',
                sifre_hash=generate_password_hash('admin123'),
                ad_soyad='Ana Yönetici',
                rol='super_admin',
                firma_adi='Ana Fabrika'
            )
            session.add(admin)
            session.commit()
            session.refresh(admin)
            logger.info(f"Varsayılan Süper Admin oluşturuldu (ID: {admin.id})")
        else:
            if not admin.rol or admin.rol == 'operator':
                admin.rol = 'super_admin'
            if not admin.firma_adi:
                admin.firma_adi = 'Ana Fabrika'
            session.commit()
            logger.info(f"Mevcut admin kullanıcısı güncellendi (ID: {admin.id})")

        # Assign all unassigned workers to default admin/patron (Option 2 requirement!)
        unassigned_count = session.query(Worker).filter((Worker.patron_id == None) | (Worker.patron_id == 0)).update({Worker.patron_id: admin.id})
        session.commit()
        logger.info(f"SQLite: {unassigned_count} adet sahipsiz çalışan 'Ana Fabrika' (Admin ID: {admin.id}) hesabına bağlandı.")

    # 2. PostgreSQL Migrasyonu (Eğer aktifse)
    try:
        from web.app import config
        merkezi_cfg = config.get('merkezi_db', {})
        pg_engine = pg_baglan(merkezi_cfg)
        if pg_engine:
            Base.metadata.create_all(bind=pg_engine)
            with pg_engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE workers ADD COLUMN IF NOT EXISTS patron_id INTEGER REFERENCES users(id) ON DELETE SET NULL"))
                    conn.commit()
                except Exception:
                    pass
                try:
                    conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS firma_adi VARCHAR(150)"))
                    conn.commit()
                except Exception:
                    pass
            logger.info("PostgreSQL veritabanı patron tabloları başarıyla güncellendi.")
    except Exception as e:
        logger.warning(f"PostgreSQL migrasyon uyarısı: {e}")

    logger.info("Patron/Firma veritabanı migrasyonu tamamlandı!")

if __name__ == '__main__':
    migrate()
