"""
İşçi Takip Sistemi - Veritabanı Yöneticisi (DatabaseManager)
SQLAlchemy 2.0 ORM Engine ve Session Yönetimi
"""

import os
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

logger = logging.getLogger(__name__)

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Sadece SQLite veritabanlarında Foreign Key, WAL modu ve synchronous=NORMAL ayarlarını aktif eder."""
    try:
        mod = getattr(type(dbapi_connection), '__module__', '')
        if mod.startswith(('sqlite3', '_sqlite3')) or 'sqlite' in mod.lower():
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
    except Exception as e:
        logger.warning(f"SQLite PRAGMA ayarlari uygulanamadi: {e}")

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = BASE_DIR / 'isci_takip.db'


class DatabaseManager:
    """
    Code-First ORM veritabanı motorunu (engine) ve oturumlarını (session)
    yöneten sınıf yapısı.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(DEFAULT_DB_PATH)
        
        # Absolute path format for SQLite engine
        db_path_obj = Path(db_path).resolve()
        self.db_url = f"sqlite:///{db_path_obj}"
        
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            echo=False
        )
        self.SessionFactory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.init_db()

    def init_db(self):
        """Code-First: Tüm veritabanı tablolarını ORM sınıflarından otomatik oluşturur."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info(f"Code-First ORM veritabanı başlatıldı: {self.db_url}")
        except Exception as e:
            logger.error(f"Veritabanı başlatma hatası: {e}")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Güvenli ORM Oturumu (Session) sağlayan bağlam yöneticisi."""
        session: Session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"ORM Oturum Hatası (Rollback yapıldı): {e}")
            raise
        finally:
            session.close()


# Proje genelinde kullanılacak varsayılan veritabanı yöneticisi örneği
db_manager = DatabaseManager()
