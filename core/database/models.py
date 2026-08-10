"""
İşçi Takip Sistemi - Code-First ORM Sınıf Modelleri (Models)
"""

import datetime
from typing import Optional, List
from sqlalchemy import (
    Integer, String, Float, Text, LargeBinary, ForeignKey, DateTime
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship
)


class Base(DeclarativeBase):
    """Tüm ORM modelleri için temel bildirim sınıfı."""
    pass


class Worker(Base):
    """Çalışan/İşçi ORM Modeli"""
    __tablename__ = 'workers'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ad: Mapped[str] = mapped_column(String(100), nullable=False)
    soyad: Mapped[str] = mapped_column(String(100), nullable=False)
    sicil_no: Mapped[Optional[str]] = mapped_column(String(50), unique=True, nullable=True)
    departman: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    istasyon_adi: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    fotograf_yolu: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    encoding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    kayit_tarihi: Mapped[str] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    aktif: Mapped[int] = mapped_column(Integer, default=1)
    patron_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # İlişkiler (Relationships) - FK ondelete kuralları ile tam uyumlu
    patron: Mapped[Optional["User"]] = relationship("User", back_populates="workers")
    durum_kayitlari: Mapped[List["DurumKaydi"]] = relationship(
        "DurumKaydi", back_populates="worker", passive_deletes=True
    )
    tespit_kayitlari: Mapped[List["TespitKaydi"]] = relationship(
        "TespitKaydi", back_populates="worker", cascade="all, delete-orphan", passive_deletes=True
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'ad': self.ad,
            'soyad': self.soyad,
            'sicil_no': self.sicil_no or '',
            'departman': self.departman or '',
            'istasyon_adi': self.istasyon_adi or '',
            'fotograf_yolu': self.fotograf_yolu or '',
            'kayit_tarihi': self.kayit_tarihi,
            'aktif': self.aktif,
            'patron_id': self.patron_id
        }


class DurumKaydi(Base):
    """İstasyon İşçi Durum Kaydı ORM Modeli"""
    __tablename__ = 'durum_kayitlari'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    istasyon_adi: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    zaman: Mapped[str] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    durum: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    worker_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workers.id", ondelete="SET NULL"), nullable=True
    )
    worker_adi: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    gonderildi: Mapped[int] = mapped_column(Integer, default=0)

    # İlişki (Relationship)
    worker: Mapped[Optional[Worker]] = relationship("Worker", back_populates="durum_kayitlari")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'istasyon_adi': self.istasyon_adi,
            'zaman': self.zaman,
            'durum': self.durum,
            'worker_id': self.worker_id,
            'worker_adi': self.worker_adi,
            'gonderildi': self.gonderildi
        }


class Alarm(Base):
    """Sistem Alarmları ORM Modeli"""
    __tablename__ = 'alarmlar'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    istasyon_adi: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    alarm_turu: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    aciklama: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    zaman: Mapped[str] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    okundu: Mapped[int] = mapped_column(Integer, default=0)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'istasyon_adi': self.istasyon_adi,
            'alarm_turu': self.alarm_turu,
            'aciklama': self.aciklama,
            'zaman': self.zaman,
            'okundu': self.okundu
        }


class TespitKaydi(Base):
    """Kamera Yüz/Nesne Tespit Kaydı ORM Modeli"""
    __tablename__ = 'tespit_kayitlari'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    worker_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workers.id", ondelete="CASCADE"), nullable=True
    )
    zaman: Mapped[str] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    istasyon: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # İlişki (Relationship)
    worker: Mapped[Optional[Worker]] = relationship("Worker", back_populates="tespit_kayitlari")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'worker_id': self.worker_id,
            'zaman': self.zaman,
            'confidence': self.confidence,
            'istasyon': self.istasyon
        }


class User(Base):
    """Kullanıcı Yönetimi ORM Modeli"""
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kullanici_adi: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    sifre_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    ad_soyad: Mapped[str] = mapped_column(String(150), nullable=False)
    rol: Mapped[str] = mapped_column(String(50), default='patron')
    firma_adi: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    istasyonlar: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    kayit_tarihi: Mapped[str] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # İlişkiler
    workers: Mapped[List["Worker"]] = relationship("Worker", back_populates="patron")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'kullanici_adi': self.kullanici_adi,
            'ad_soyad': self.ad_soyad,
            'rol': self.rol,
            'firma_adi': self.firma_adi or '',
            'istasyonlar': self.istasyonlar or '',
            'kayit_tarihi': self.kayit_tarihi
        }


class GunlukOzet(Base):
    """Günlük İstasyon Özet Kaydı ORM Modeli"""
    __tablename__ = 'gunluk_ozetler'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tarih: Mapped[str] = mapped_column(String(50), nullable=False)
    istasyon_adi: Mapped[str] = mapped_column(String(100), nullable=False)
    worker_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    worker_adi: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    toplam_aktif_sn: Mapped[int] = mapped_column(Integer, default=0)
    toplam_kaynak_sn: Mapped[int] = mapped_column(Integer, default=0)
    toplam_inaktif_sn: Mapped[int] = mapped_column(Integer, default=0)
    toplam_telefon_sn: Mapped[int] = mapped_column(Integer, default=0)

    telefon_ihlal_sayisi: Mapped[int] = mapped_column(Integer, default=0)
    son_guncelleme: Mapped[Optional[str]] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


class Camera(Base):
    """Pi 5 / İstasyon Kamera Yönetimi ORM Modeli"""
    __tablename__ = 'cameras'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    istasyon_adi: Mapped[str] = mapped_column(String(100), nullable=False)
    ip_adresi: Mapped[str] = mapped_column(String(100), nullable=False)
    patron_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    patron_adi: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    aktif: Mapped[int] = mapped_column(Integer, default=1)
    kayit_tarihi: Mapped[str] = mapped_column(
        String(50),
        default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    patron: Mapped[Optional["User"]] = relationship("User")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'istasyon_adi': self.istasyon_adi,
            'ip_adresi': self.ip_adresi,
            'patron_id': self.patron_id,
            'patron_adi': self.patron_adi or '',
            'aktif': self.aktif,
            'kayit_tarihi': self.kayit_tarihi
        }

