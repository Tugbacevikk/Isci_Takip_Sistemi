import os
import time
import logging
import datetime
import threading
from typing import Optional

from sqlalchemy import create_engine, select, update, delete, UniqueConstraint, String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.database.models import DurumKaydi, Alarm, TespitKaydi
from core.database.connection import DatabaseManager, db_manager

logger = logging.getLogger(__name__)


# ── Code-First ORM Modeli (Merkezi PostgreSQL Tablosu) ─────────────────────
class CentralBase(DeclarativeBase):
    """Merkezi PostgreSQL ORM Temel Sınıfı"""
    pass


class CentralDurumKaydiModel(CentralBase):
    """Merkezi PostgreSQL durum_kayitlari ORM Modeli (Code-First)"""
    __tablename__ = 'durum_kayitlari'
    __table_args__ = (
        UniqueConstraint('istasyon_adi', 'kaynak_id', name='uq_istasyon_kaynak'),
        {'extend_existing': True}
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    istasyon_adi: Mapped[str] = mapped_column(String(100), nullable=False)
    zaman: Mapped[str] = mapped_column(String(50), nullable=False)
    durum: Mapped[str] = mapped_column(String(100), nullable=False)
    kaynak_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    worker_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    worker_adi: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    olusturulma: Mapped[Optional[str]] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def pg_baglan(cfg: dict):
    """
    Merkezi PostgreSQL veritabanı bağlantısı oluşturur ve SQLAlchemy Engine nesnesi döndürür.
    """
    env_host = os.getenv('POSTGRES_HOST')
    env_user = os.getenv('POSTGRES_USER')
    env_pass = os.getenv('POSTGRES_PASSWORD')
    env_db   = os.getenv('POSTGRES_DB')
    env_port = os.getenv('POSTGRES_PORT')

    host = cfg.get('host') or cfg.get('postgres_host')
    if not host or host == '127.0.0.1':
        host = env_host or host or '127.0.0.1'

    port = int(cfg.get('port') or cfg.get('postgres_port') or env_port or 5432)
    dbname = cfg.get('dbname') or cfg.get('postgres_db') or cfg.get('db') or env_db or 'fabrika_takip'

    user = cfg.get('kullanici') or cfg.get('user') or cfg.get('postgres_user')
    if not user or user == 'postgres':
        user = env_user or user or 'postgres'

    password = cfg.get('sifre') if cfg.get('sifre') is not None else cfg.get('password')
    if password is None or password == '':
        password = env_pass or cfg.get('postgres_password', '')

    pg_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    try:
        engine = create_engine(
            pg_url,
            connect_args={"connect_timeout": 5},
            pool_pre_ping=True,
            isolation_level="AUTOCOMMIT",
            use_native_hstore=False,
            echo=False
        )
        with engine.connect() as conn:
            pass
        logger.info(f"[PG] Merkezi PostgreSQL'e bağlandı -> {host}:{port}/{dbname}")
        return engine
    except Exception as e:
        logger.warning(f"[PG] Bağlantı hatası: {e}")
        return None


def pg_baglantiyi_kapat(engine):
    """SQLAlchemy motorunu ve bağlantı havuzunu kapatır."""
    if engine is not None:
        try:
            engine.dispose()
        except Exception as e:
            logger.debug(f"[PG] Engine dispose hatası: {e}")


def pg_tablo_hazirla(engine) -> bool:
    """Code-First: PostgreSQL'de hedef tablo yoksa ORM Metadata ile otomatik oluşturur."""
    if engine is None:
        return False

    try:
        CentralBase.metadata.create_all(engine)
        logger.info("[PG] Merkezi PostgreSQL Code-First veritabanı ve tabloları doğrulandı.")
        return True
    except Exception as e:
        logger.error(f"[PG] Tablo kontrolü/oluşturma hatası: {e}")
        return False


def senkronize_et(db_mgr: DatabaseManager, engine, istasyon_adi: str) -> int:
    """
    SQLite'taki gonderildi=0 kayıtları ORM ile çeker ve PostgreSQL'e
    Code-First ORM (pg_insert on_conflict_do_nothing) ile aktarır.
    """
    if engine is None or db_mgr is None:
        return 0

    # 1. Yerel SQLite'tan gönderilmeyen kayıtları oku
    values_list = []
    local_ids = []
    try:
        with db_mgr.get_session() as local_session:
            stmt = select(DurumKaydi).where(DurumKaydi.gonderildi == 0).order_by(DurumKaydi.id.asc()).limit(200)
            kayitlar = local_session.scalars(stmt).all()
            if not kayitlar:
                return 0
            for r in kayitlar:
                st_name = r.istasyon_adi or istasyon_adi
                if not st_name or st_name == 'auto':
                    import socket
                    st_name = socket.gethostname()

                values_list.append({
                    'istasyon_adi': st_name,
                    'zaman': str(r.zaman),
                    'durum': r.durum,
                    'kaynak_id': r.id,
                    'worker_id': r.worker_id,
                    'worker_adi': r.worker_adi
                })
                local_ids.append(r.id)
    except Exception as e:
        logger.error(f"[PG] Yerel SQLite okuma hatası: {e}")
        return 0

    if not values_list:
        return 0

    # 2. Merkezi PostgreSQL'e Code-First ORM pg_insert ile aktar
    inserted_count = 0
    try:
        insert_stmt = pg_insert(CentralDurumKaydiModel).values(values_list)
        upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=['istasyon_adi', 'kaynak_id'])
        with Session(engine) as pg_session:
            res = pg_session.execute(upsert_stmt)
            pg_session.commit()
            inserted_count = res.rowcount if (res and res.rowcount is not None) else len(local_ids)
    except Exception as e:
        logger.error(f"[PG] Merkezi PostgreSQL yazma hatası: {e}")
        return -1

    # 3. Başarılı ise yerel SQLite tarafında gonderildi = 1 yap
    try:
        with db_mgr.get_session() as local_session:
            local_session.execute(
                update(DurumKaydi)
                .where(DurumKaydi.id.in_(local_ids))
                .values(gonderildi=1)
            )
        if inserted_count > 0:
            logger.info(f"[PG] {inserted_count} adet yeni kayıt başarıyla PostgreSQL'e aktarıldı.")
        else:
            logger.warning(f"[PG] {len(local_ids)} kayıt işlendi ancak çakışma (on_conflict) nedeniyle PostgreSQL'e yeni satır eklenmedi. Lütfen bilgisayarın 'config.yaml' dosyasında 'istasyon_adi' değerinin benzersiz olduğundan emin olun.")
        return len(local_ids)
    except Exception as e:
        logger.error(f"[PG] Yerel SQLite gonderildi güncelleme hatası: {e}")
        return 0


def veritabanlarini_temizle(
    db_mgr: DatabaseManager,
    engine,
    local_retention_days: int = 7,
    pg_retention_days: int = 30
) -> dict:
    """
    1. Yerel SQLite: PostgreSQL'e aktarılmış (gonderildi=1) ve local_retention_days (7 gün) günden eski kayıtları siler.
    2. Merkezi PostgreSQL: pg_retention_days (30 gün) günden eski ham durum kayıtlarını siler.
    """
    res = {'local_deleted': 0, 'pg_deleted': 0}
    now = datetime.datetime.now()

    # 1. Yerel SQLite Temizliği
    if db_mgr:
        try:
            local_cutoff = (now - datetime.timedelta(days=local_retention_days)).strftime("%Y-%m-%d %H:%M:%S")
            alarm_cutoff = (now - datetime.timedelta(days=pg_retention_days)).strftime("%Y-%m-%d %H:%M:%S")
            with db_mgr.get_session() as local_session:
                stmt_durum = delete(DurumKaydi).where(
                    DurumKaydi.gonderildi == 1,
                    DurumKaydi.zaman < local_cutoff
                )
                r_durum = local_session.execute(stmt_durum)

                stmt_alarm = delete(Alarm).where(Alarm.zaman < alarm_cutoff)
                r_alarm = local_session.execute(stmt_alarm)

                stmt_tespit = delete(TespitKaydi).where(TespitKaydi.zaman < alarm_cutoff)
                r_tespit = local_session.execute(stmt_tespit)

                res['local_deleted'] = (r_durum.rowcount or 0) + (r_alarm.rowcount or 0) + (r_tespit.rowcount or 0)
                if res['local_deleted'] > 0:
                    logger.info(f"[TEMİZLİK] Yerel SQLite'tan {res['local_deleted']} eski kayıt silindi (>{local_retention_days} gün).")
        except Exception as e:
            logger.error(f"[TEMİZLİK] Yerel SQLite temizleme hatası: {e}")

    # 2. Merkezi PostgreSQL Temizliği
    if engine:
        try:
            pg_cutoff = (now - datetime.timedelta(days=pg_retention_days)).strftime("%Y-%m-%d %H:%M:%S")
            with Session(engine) as pg_session:
                stmt_pg = delete(CentralDurumKaydiModel).where(CentralDurumKaydiModel.zaman < pg_cutoff)
                r_pg = pg_session.execute(stmt_pg)
                pg_session.commit()
                res['pg_deleted'] = r_pg.rowcount or 0
                if res['pg_deleted'] > 0:
                    logger.info(f"[TEMİZLİK] Merkezi PostgreSQL'den {res['pg_deleted']} eski kayıt silindi (>{pg_retention_days} gün).")
        except Exception as e:
            logger.error(f"[TEMİZLİK] Merkezi PostgreSQL temizleme hatası: {e}")

    return res


class SenkronThread(threading.Thread):
    """
    Belirtilen aralıkta SQLite -> PostgreSQL senkronizasyonu ve periyodik veritabanı temizliği yapan daemon thread.
    """

    def __init__(self, db_mgr: DatabaseManager, merkezi_db_cfg: dict, istasyon_adi: str = "auto"):
        super().__init__(name="PgSenkronThread", daemon=True)
        self._db_mgr = db_mgr or db_manager
        self._cfg = merkezi_db_cfg or {}
        self._istasyon_adi = istasyon_adi
        self._aralik_sn = int(self._cfg.get('senkron_araligi_sn', 60))
        self._local_retention_days = int(self._cfg.get('local_retention_days', 7))
        self._pg_retention_days = int(self._cfg.get('pg_retention_days', 30))
        self._durdurma_olayi = threading.Event()
        self._pg_engine = None
        self._last_cleanup_time = 0.0

    def durdur(self):
        """Thread'e durma sinyali gönder."""
        self._durdurma_olayi.set()

    def _yeniden_baglan(self):
        """Mevcut bağlantıyı kapatıp yenisini açar."""
        pg_baglantiyi_kapat(self._pg_engine)
        self._pg_engine = pg_baglan(self._cfg)
        if self._pg_engine:
            pg_tablo_hazirla(self._pg_engine)

    def run(self):
        logger.info(f"[PG] Senkronizasyon thread'i başlatıldı (aralık: {self._aralik_sn}s, istasyon: {self._istasyon_adi})")
        self._yeniden_baglan()

        while not self._durdurma_olayi.is_set():
            if self._pg_engine is None:
                self._yeniden_baglan()

            if self._pg_engine:
                gonderilen = senkronize_et(self._db_mgr, self._pg_engine, self._istasyon_adi)
                if gonderilen == -1:
                    self._pg_engine = None
                elif gonderilen > 0:
                    simdi_str = datetime.datetime.now().strftime("%H:%M:%S")
                    logger.info(f"[PG] {gonderilen} kayıt merkezi DB'ye gönderildi ({simdi_str})")

            # Periyodik Temizlik Check (Günde 1 kez / 24 saatte bir)
            now_time = time.time()
            if now_time - self._last_cleanup_time >= 86400 or self._last_cleanup_time == 0.0:
                self._last_cleanup_time = now_time
                veritabanlarini_temizle(
                    self._db_mgr,
                    self._pg_engine,
                    local_retention_days=self._local_retention_days,
                    pg_retention_days=self._pg_retention_days
                )

            self._durdurma_olayi.wait(timeout=self._aralik_sn)

        # Son çıkış senkronizasyonu (Kapanırken kalan tüm kayıtları PostgreSQL'e aktar)
        if self._pg_engine:
            try:
                gonderilen = senkronize_et(self._db_mgr, self._pg_engine, self._istasyon_adi)
                if gonderilen > 0:
                    logger.info(f"[PG] Kapanış senkronizasyonu: {gonderilen} kayıt merkezi DB'ye aktarıldı.")
            except Exception as e:
                logger.error(f"[PG] Kapanış senkronizasyon hatası: {e}")

        pg_baglantiyi_kapat(self._pg_engine)
        logger.info("[PG] Senkronizasyon thread'i durduruldu.")
