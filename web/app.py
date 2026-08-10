"""
İşçi Takip Sistemi - Web Arayüzü (SQLAlchemy Code-First ORM Mimarisi)
Flask + Flask-SocketIO tabanlı yönetim paneli
Kullanım: python web/app.py
Tarayıcı: http://localhost:5000
"""

import os
import sys

# Windows terminal UTF-8 sorunu çözümü
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import json
import time
import logging
import datetime
import threading
from functools import wraps
from typing import Optional, List, Dict, Any
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import yaml

from flask import (
    Flask, render_template, request, jsonify,
    Response, redirect, url_for,
    session, flash,
)
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

try:
    import ultralytics
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# ---------------------------------------------------------------------------
# Yol ayarları ve Core modülleri
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent  # proje kökü
CORE_DIR   = BASE_DIR / 'core'
WEB_DIR    = Path(__file__).parent
DB_PATH    = BASE_DIR / 'isci_takip.db'
CONFIG_PATH = BASE_DIR / 'config.yaml'
PHOTOS_DIR  = WEB_DIR / 'static' / 'workers'

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass

for _p in [str(CORE_DIR), str(BASE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# SQLAlchemy 2.0 ORM & Core modülleri
from sqlalchemy import select, delete, func, case, and_, or_, desc, String
from core.database.models import Worker, DurumKaydi, Alarm, User, Camera
from core.database.connection import db_manager
from camera_manager import CameraProcessor

try:
    from pg_sync import SenkronThread, veritabanlarini_temizle
    HAS_PG_SYNC = True
except ImportError:
    HAS_PG_SYNC = False
    SenkronThread = None
    veritabanlarini_temizle = None

# ---------------------------------------------------------------------------
# Loglama - UTF-8 handler + 5 MB RotatingFileHandler
# ---------------------------------------------------------------------------
import io as _io
from logging.handlers import RotatingFileHandler

LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

_log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

_stream_handler = logging.StreamHandler(
    _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stdout, 'buffer') else sys.stdout
)
_stream_handler.setFormatter(_log_formatter)

_file_handler = RotatingFileHandler(
    LOGS_DIR / 'web.log',
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_stream_handler, _file_handler],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask & SocketIO
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(WEB_DIR / 'templates'),
    static_folder=str(WEB_DIR / 'static'),
)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'isci-takip-secret-2024-xK9mP2')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': 'Yüklenen video dosyası çok büyük! (Maksimum 500 MB yükleyebilirsiniz.)'}), 413
app.config['SESSION_PERMANENT'] = False

if HAS_CORS:
    CORS(app)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# ---------------------------------------------------------------------------
# Global değişkenler
# ---------------------------------------------------------------------------
camera_processor: CameraProcessor = None
camera_thread: threading.Thread = None
face_recognizer = None
config: dict = {}
last_status: dict = {
    'durum': 'Kamera Başlatılmadı',
    'renk': '#888888',
    'fps': 0.0,
    'kisi_sayisi': 0,
    'istasyon': 'N/A',
    'zaman': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'running': False,
    'worker_name': '',
    'worker_confidence': 0.0,
    'phone_detected': False,
    'camera_id': '',
}

# ---------------------------------------------------------------------------
# Veritabanı İlklendirme (Code-First ORM)
# ---------------------------------------------------------------------------

def init_db():
    """Tüm ORM modellerini (Code-First) oluşturur ve varsayılan yöneticiyi ekler."""
    db_manager.init_db()
    try:
        with db_manager.get_session() as session_orm:
            admin_user = session_orm.scalars(select(User).where(User.kullanici_adi == 'admin')).first()
            if not admin_user:
                default_admin = User(
                    kullanici_adi='admin',
                    sifre_hash=generate_password_hash('admin123'),
                    ad_soyad='Sözleşmeli Yönetici',
                    rol='admin'
                )
                session_orm.add(default_admin)
                session_orm.commit()
                logger.info("Varsayılan admin kullanıcısı oluşturuldu (admin / admin123).")
            elif admin_user.rol != 'admin':
                admin_user.rol = 'admin'
                session_orm.commit()
        logger.info(f"ORM Veritabanı başlatıldı: {DB_PATH}")
    except Exception as e:
        logger.error(f"Veritabanı ilklendirme hatası: {e}")

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    'camera_id': 0,
    'camera_width': 1280,
    'camera_height': 720,
    'camera_fps': 30,
    'istasyon_adi': 'auto',
    'roi_x1': 0.12,
    'roi_y1': 0.24,
    'roi_x2': 0.75,
    'roi_y2': 0.85,
    'save_interval': 1,
    'merkezi_db': {
        'aktif': os.getenv('POSTGRES_ENABLED', 'true').lower() in ('true', '1'),
        'host': os.getenv('POSTGRES_HOST', '192.168.30.222'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'dbname': os.getenv('POSTGRES_DB', 'fabrika_takip'),
        'kullanici': os.getenv('POSTGRES_USER', 'takip_user'),
        'sifre': os.getenv('POSTGRES_PASSWORD', 'admin123'),
        'senkron_araligi_sn': int(os.getenv('POSTGRES_SYNC_INTERVAL', 5)),
        'local_retention_days': int(os.getenv('LOCAL_RETENTION_DAYS', 7)),
        'pg_retention_days': int(os.getenv('PG_RETENTION_DAYS', 30)),
    }
}


def load_config() -> dict:
    """config.yaml dosyasını okur, yoksa varsayılanı yazar."""
    global config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        merged_merkezi = {**_DEFAULT_CONFIG.get('merkezi_db', {}), **(loaded.get('merkezi_db') or {})}
        config = {**_DEFAULT_CONFIG, **loaded, 'merkezi_db': merged_merkezi}
    else:
        config = dict(_DEFAULT_CONFIG)
        save_config(config)
    return config


def save_config(cfg: dict):
    """Yapılandırmayı config.yaml dosyasına kaydeder."""
    global config
    config = cfg
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    logger.info("Yapılandırma kaydedildi.")

# ---------------------------------------------------------------------------
# Kamera tarama & Yardımcılar
# ---------------------------------------------------------------------------

def get_camera_device_names() -> list:
    """Windows DirectShow veya Linux V4L2 üzerinden bağlı kamera cihaz isimlerini alır."""
    try:
        return CameraProcessor.get_camera_device_names()
    except Exception:
        return []


def scan_cameras(max_index: int = 5) -> list:
    """Kullanılabilir kameraları tarar ve cihaz detayları listesini döndürür."""
    try:
        return CameraProcessor.scan_cameras(max_index=max_index)
    except Exception as e:
        logger.debug(f"Kamera tarama hatası: {e}")
        return []


def _get_dark_frame() -> bytes:
    """'Kamera Başlatılmadı' yazılı koyu kare döndürür."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    cv2.putText(
        frame, "Kamera Baslatilmadi", (120, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA,
    )
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()


def generate_frames():
    """MJPEG akış üreteci (CameraProcessor.get_current_frame() kullanır)."""
    global camera_processor, last_status
    dark_frame = _get_dark_frame()

    try:
        while True:
            if (
                camera_processor is not None
                and camera_processor.is_running
            ):
                frame = camera_processor.get_current_frame()
                if frame is not None:
                    cur_st = camera_processor.get_status()
                    last_status.update(cur_st)
                    last_status['running'] = True

                    if frame.shape[1] > 640:
                        target_w = 640
                        target_h = int(640 * frame.shape[0] / frame.shape[1])
                        encode_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        encode_frame = frame

                    _, jpeg = cv2.imencode('.jpg', encode_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    frame_bytes = jpeg.tobytes()

                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                    time.sleep(0.005)
                else:
                    time.sleep(0.01)
            else:
                last_status['running'] = False
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + dark_frame + b'\r\n'
                )
                time.sleep(0.2)
    except (GeneratorExit, ConnectionResetError, BrokenPipeError, OSError):
        pass
    except Exception as e:
        logger.debug(f"Kamera akış üreteç sonlandı: {e}")



def _broadcast_status():
    """Her saniye durum güncellemesi yayınlar."""
    global last_status
    while True:
        try:
            st = _get_current_status()
            socketio.emit('status_update', st)
        except Exception:
            pass
        time.sleep(1.0)


def _get_current_status() -> dict:
    global camera_processor, last_status
    if camera_processor is not None and camera_processor.is_running:
        st = camera_processor.get_current_status()
        last_status.update(st)
        last_status['running'] = True
    else:
        last_status['running'] = False
        last_status['durum'] = 'Kamera Kapalı'
        last_status['status'] = 'Kamera Kapalı'
        last_status['worker_name'] = ''
        last_status['worker_confidence'] = 0.0
        last_status['kisi_sayisi'] = 0
        last_status['person_count'] = 0
        last_status['fps'] = 0.0
    return last_status

# ---------------------------------------------------------------------------
# Oturum & Yetkilendirme Dekoratörleri
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Giriş yapmalısınız.'}), 401
            flash('Bu sayfayı görüntülemek için giriş yapmalısınız.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Giriş yapmalısınız.'}), 401
            flash('Bu sayfayı görüntülemek için giriş yapmalısınız.', 'warning')
            return redirect(url_for('login', next=request.url))
        if session.get('role') not in ('admin', 'super_admin'):
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Yönetici yetkisi gereklidir.'}), 403
            flash('Bu işlem için yönetici yetkisi gereklidir.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_user():
    """Tüm şablonlara mevcut kullanıcı bilgisini aktarır."""
    if 'user_id' in session:
        return {
            'current_user': {
                'id': session.get('user_id'),
                'username': session.get('username'),
                'full_name': session.get('full_name'),
                'rol': session.get('role'),
            }
        }
    return {'current_user': None}

# ---------------------------------------------------------------------------
# Kimlik Doğrulama Rotaları (ORM Nesnel)
# ---------------------------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
        return render_template('login.html')

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()

    if not username or not password:
        flash('Kullanıcı adı ve şifre gereklidir.', 'danger')
        return render_template('login.html')

    try:
        with db_manager.get_session() as session_orm:
            user = session_orm.scalars(select(User).where(User.kullanici_adi == username)).first()

            if user and check_password_hash(user.sifre_hash, password):
                session.pop('_flashes', None)
                session['user_id'] = user.id
                session['username'] = user.kullanici_adi
                session['full_name'] = user.ad_soyad
                session['role'] = user.rol
                session['firma_adi'] = user.firma_adi or ''
                flash(f'Hoş geldiniz, {user.ad_soyad}!', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Geçersiz kullanıcı adı veya şifre.', 'danger')
                return render_template('login.html')
    except Exception as e:
        logger.error(f"Giriş hatası (ORM): {e}")
        flash('Veritabanı hatası oluştu.', 'danger')
        return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Başarıyla çıkış yapıldı.', 'info')
    return redirect(url_for('dashboard'))


def get_current_patron_access():
    """
    Oturum açan kullanıcının (patron_id, is_super_admin, patron_stations_list) bilgisini döndürür.
    Süper Admin / Admin ise: (None, True, []) -> Tüm sistem yetkisi var
    Patron ise: (user_id, False, ['Istasyon-A', 'Istasyon-B']) -> Sadece kendi veya atanan istasyon verilerine erişebilir
    """
    user_id = session.get('user_id')
    if not user_id:
        return -99999, False, []
    role = session.get('role', 'operator')
    if role in ('super_admin', 'admin'):
        return None, True, []

    stations = []
    try:
        with db_manager.get_session() as session_orm:
            u = session_orm.get(User, user_id)
            if u and u.istasyonlar:
                stations = [s.strip() for s in u.istasyonlar.split(',') if s.strip()]
    except Exception:
        pass

    return user_id, False, stations


def get_current_patron_id():
    p_id, is_super, _ = get_current_patron_access()
    return p_id, is_super

# ---------------------------------------------------------------------------
# Kullanıcı Yönetimi API (ORM Nesnel + Aliases)
# ---------------------------------------------------------------------------

@app.route('/api/patrons', methods=['GET'])
@app.route('/api/patrons/list', methods=['GET'])
@admin_required
def api_patrons_list():
    try:
        with db_manager.get_session() as session_orm:
            include_all = request.args.get('all', 'false').lower() in ('true', '1')
            if include_all:
                users = session_orm.scalars(select(User).order_by(User.id.asc())).all()
            else:
                users = session_orm.scalars(
                    select(User).where(User.rol == 'patron').order_by(User.id.asc())
                ).all()
            patrons = [u.to_dict() for u in users]
            return jsonify({'success': True, 'patrons': patrons, 'users': patrons})
    except Exception as e:
        logger.error(f"Patron listesi hatası: {e}")
        return jsonify({'success': False, 'patrons': []}), 500


@app.route('/api/users', methods=['GET'])
@app.route('/api/users/list', methods=['GET'])
@admin_required
def api_users_list():
    try:
        with db_manager.get_session() as session_orm:
            users = session_orm.scalars(select(User).order_by(User.id.asc())).all()
            user_list = [u.to_dict() for u in users]
            return jsonify({'success': True, 'users': user_list, 'data': user_list})
    except Exception as e:
        logger.error(f"Kullanıcı listesi hatası: {e}")
        return jsonify({'success': False, 'error': str(e), 'users': []}), 500


@app.route('/api/users', methods=['POST'])
@app.route('/api/users/add', methods=['POST'])
@admin_required
def api_users_add():
    data = request.get_json() or {}
    username = (data.get('kullanici_adi') or data.get('username') or '').strip()
    password = (data.get('sifre') or data.get('password') or '').strip()
    fullname = (data.get('ad_soyad') or data.get('fullname') or '').strip()
    role = (data.get('rol') or data.get('role') or 'patron').strip()
    firma_adi = (data.get('firma_adi') or data.get('company') or 'Fabrika').strip()
    istasyonlar = (data.get('istasyonlar') or data.get('stations') or '').strip()

    if not username or not password or not fullname:
        return jsonify({'success': False, 'message': 'Kullanıcı adı, ad soyad ve şifre gereklidir.'}), 400

    try:
        with db_manager.get_session() as session_orm:
            existing = session_orm.scalars(select(User).where(User.kullanici_adi == username)).first()
            if existing:
                return jsonify({'success': False, 'message': 'Bu kullanıcı adı zaten mevcut.'}), 400

            new_user = User(
                kullanici_adi=username,
                sifre_hash=generate_password_hash(password),
                ad_soyad=fullname,
                rol=role,
                firma_adi=firma_adi or None,
                istasyonlar=istasyonlar or None
            )
            session_orm.add(new_user)
            session_orm.commit()
        return jsonify({'success': True, 'message': f'"{username}" kullanıcısı/patronu başarıyla eklendi.'})
    except Exception as e:
        logger.error(f"Kullanıcı ekleme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['DELETE', 'POST'])
@app.route('/api/users/<int:user_id>/delete', methods=['DELETE', 'POST'])
@admin_required
def api_users_delete(user_id):
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'message': 'Kendi hesabınızı silemezsiniz.'}), 400

    try:
        with db_manager.get_session() as session_orm:
            user = session_orm.get(User, user_id)
            if user:
                session_orm.delete(user)
                session_orm.commit()
                return jsonify({'success': True, 'message': 'Kullanıcı başarıyla silindi.'})
            return jsonify({'success': False, 'message': 'Kullanıcı bulunamadı.'}), 404
    except Exception as e:
        logger.error(f"Kullanıcı silme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/users/<int:user_id>/update', methods=['POST', 'PUT'])
@admin_required
def api_users_update(user_id):
    data = request.get_json() or {}
    fullname = (data.get('ad_soyad') or data.get('fullname') or '').strip()
    role = (data.get('rol') or data.get('role') or '').strip()
    firma_adi = (data.get('firma_adi') or data.get('company') or '').strip()
    istasyonlar = (data.get('istasyonlar') or data.get('stations') or '').strip()
    password = (data.get('sifre') or data.get('password') or '').strip()

    try:
        with db_manager.get_session() as session_orm:
            user = session_orm.get(User, user_id)
            if not user:
                return jsonify({'success': False, 'message': 'Kullanıcı bulunamadı.'}), 404

            if fullname: user.ad_soyad = fullname
            if role: user.rol = role
            if firma_adi is not None: user.firma_adi = firma_adi
            if 'istasyonlar' in data or 'stations' in data: user.istasyonlar = istasyonlar or None
            if password: user.sifre_hash = generate_password_hash(password)
            session_orm.commit()

        return jsonify({'success': True, 'message': 'Kullanıcı yetkileri ve atanan istasyonlar güncellendi.'})
    except Exception as e:
        logger.error(f"Kullanıcı güncelleme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/patrons/assign_worker', methods=['POST'])
@admin_required
def api_patrons_assign_worker():
    data = request.get_json() or {}
    worker_id = data.get('worker_id')
    patron_id = data.get('patron_id')

    if not worker_id:
        return jsonify({'success': False, 'message': 'Çalışan ID gereklidir.'}), 400

    try:
        with db_manager.get_session() as session_orm:
            worker = session_orm.get(Worker, int(worker_id))
            if not worker:
                return jsonify({'success': False, 'message': 'Çalışan bulunamadı.'}), 404

            target_p_id = int(patron_id) if (patron_id and str(patron_id).isdigit()) else None
            worker.patron_id = target_p_id

        return jsonify({'success': True, 'message': 'Çalışan patron ataması güncellendi.'})
    except Exception as e:
        logger.error(f"Patron çalışan atama hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------------------------------------------------------------------
# Sayfa Rotaları
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    cams = scan_cameras()
    return render_template('dashboard.html', cameras=cams)


@app.route('/cameras')
def cameras():
    cams = scan_cameras()
    return render_template('cameras.html', cameras=cams, config=config)


@app.route('/workers')
@app.route('/workers_page')
def workers_page():
    patron_id, is_super, stations = get_current_patron_access()
    try:
        with db_manager.get_session() as session_orm:
            stmt = select(Worker)
            if patron_id != -99999 and not is_super:
                conds = []
                if patron_id and patron_id > 0:
                    conds.append(Worker.patron_id == patron_id)
                if stations:
                    conds.append(Worker.istasyon_adi.in_(stations))
                if conds:
                    stmt = stmt.where(or_(*conds))
                else:
                    stmt = stmt.where(Worker.patron_id == patron_id)
            workers = session_orm.scalars(stmt).all()
            return render_template('workers.html', workers=[w.to_dict() for w in workers])
    except Exception as e:
        logger.error(f"Workers page hatası: {e}")
        return render_template('workers.html', workers=[])


@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html')


@app.route('/alarms')
@app.route('/alarms_page')
def alarms_page():
    return render_template('alarms.html')


@app.route('/live-cameras')
@app.route('/live_cameras')
@login_required
def live_cameras():
    patron_id, is_super, stations = get_current_patron_access()
    try:
        with db_manager.get_session() as session_orm:
            users = session_orm.scalars(select(User).where(User.rol == 'patron')).all()
            patrons = [u.to_dict() for u in users]
    except Exception:
        patrons = []
    return render_template('live_cameras.html', is_admin=is_super, patrons=patrons)


def is_camera_authorized(cam: Camera, user_id: Optional[int], is_super: bool, stations: List[str]) -> bool:
    """
    Sıkı İstasyon Bazlı Kamera Yetki Denetimi:
    - Super Admin / Admin -> TÜM KAMERALARA YETKİLİ
    - Patron -> Kameranın istasyon adı (örn: 'Istasyon-1') kullanıcının yetkili istasyonlar listesinde (User.istasyonlar) OLMAK ZORUNDADIR.
    """
    if is_super:
        return True
    if not cam or not cam.aktif:
        return False
    
    # Patron yetki kontrolü: Kameranın istasyonu kullanıcının yetkili olduğu istasyonlar listesinde var mı?
    if stations and len(stations) > 0:
        return cam.istasyon_adi in stations

    return False


def _get_unauthorized_frame() -> Response:
    """'Yetkisiz Erişim' yazılı kırmızı uyarı karesi döndürür."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 20, 35)
    cv2.putText(
        frame, "YETKISIZ ERISIM", (160, 220),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 239), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Bu kamerayi izleme yetkiniz yok", (120, 270),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/api/cameras/manage', methods=['GET'])
@login_required
def api_cameras_list():
    patron_id, is_super, stations = get_current_patron_access()
    user_id = session.get('user_id')
    try:
        with db_manager.get_session() as session_orm:
            all_cams = session_orm.scalars(select(Camera).where(Camera.aktif == 1).order_by(Camera.id.asc())).all()
            allowed_cams = [c for c in all_cams if is_camera_authorized(c, user_id, is_super, stations)]
            return jsonify({'success': True, 'cameras': [c.to_dict() for c in allowed_cams]})
    except Exception as e:
        logger.error(f"Kamera listesi getirme hatası: {e}")
        return jsonify({'success': False, 'cameras': []})


@app.route('/api/proxy_feed/<int:cam_id>')
@login_required
def api_proxy_feed(cam_id):
    """Kamera yayınını sıkı yetki kontrolünden geçirerek sunar."""
    patron_id, is_super, stations = get_current_patron_access()
    user_id = session.get('user_id')
    try:
        with db_manager.get_session() as session_orm:
            cam = session_orm.get(Camera, cam_id)
            if not cam or not cam.aktif:
                return Response(_get_dark_frame(), mimetype='image/jpeg')

            # Sıkı Yetki Kontrolü
            if not is_camera_authorized(cam, user_id, is_super, stations):
                return _get_unauthorized_frame(), 403

            ip = (cam.ip_adresi or '').strip()
            if not ip or ip in ('127.0.0.1', 'localhost', '0.0.0.0'):
                return video_feed()

            target_url = f"http://{ip}:5000/api/video_feed"
            import urllib.request

            def generate_proxy_stream():
                try:
                    req = urllib.request.urlopen(target_url, timeout=5)
                    while True:
                        chunk = req.read(4096)
                        if not chunk:
                            break
                        yield chunk
                except Exception as ex:
                    logger.debug(f"Proxy akış okuma hatası ({ip}): {ex}")
                    yield _get_dark_frame()

            return Response(generate_proxy_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Proxy feed hatası: {e}")
        return Response(_get_dark_frame(), mimetype='image/jpeg')


@app.route('/api/cameras/manage', methods=['POST'])
@admin_required
def api_cameras_add():
    data = request.get_json() or {}
    istasyon_adi = (data.get('istasyon_adi') or '').strip()
    ip_adresi = (data.get('ip_adresi') or '').strip()

    if not istasyon_adi or not ip_adresi:
        return jsonify({'success': False, 'message': 'İstasyon adı ve IP adresi gereklidir.'}), 400

    try:
        with db_manager.get_session() as session_orm:
            new_cam = Camera(
                istasyon_adi=istasyon_adi,
                ip_adresi=ip_adresi,
                patron_id=None,
                patron_adi=None,
                aktif=1
            )
            session_orm.add(new_cam)
            session_orm.commit()
            return jsonify({'success': True, 'message': f'"{istasyon_adi}" kamerasını/istasyonunu başarıyla eklendi.', 'camera': new_cam.to_dict()})
    except Exception as e:
        logger.error(f"Kamera ekleme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/manage/<int:cam_id>', methods=['DELETE', 'POST'])
@admin_required
def api_cameras_delete(cam_id):
    try:
        with db_manager.get_session() as session_orm:
            cam = session_orm.get(Camera, cam_id)
            if not cam:
                return jsonify({'success': False, 'message': 'Kamera bulunamadı.'}), 404
            session_orm.delete(cam)
            session_orm.commit()
            return jsonify({'success': True, 'message': 'Kamera başarıyla silindi.'})
    except Exception as e:
        logger.error(f"Kamera silme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/settings')
@admin_required
def settings():
    try:
        with db_manager.get_session() as session_orm:
            users = session_orm.scalars(select(User).order_by(User.id.asc())).all()
            user_list = [u.to_dict() for u in users]
    except Exception as e:
        logger.error(f"Ayarlar kullanıcı listesi okuma hatası: {e}")
        user_list = []
    return render_template('settings.html', config=config, users=user_list)

# ---------------------------------------------------------------------------
# API Rotaları (ORM Nesnel Sorgular + Aliases)
# ---------------------------------------------------------------------------

@app.route('/api/video_feed')
@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/status', methods=['GET'])
@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(_get_current_status())


@app.route('/api/system/info', methods=['GET'])
@app.route('/api/system_info', methods=['GET'])
def api_system_info():
    try:
        import platform
        import psutil
        db_size_str = "0 MB"
        if DB_PATH.exists():
            bytes_size = DB_PATH.stat().st_size
            if bytes_size > 1024 * 1024:
                db_size_str = f"{bytes_size / (1024 * 1024):.2f} MB"
            else:
                db_size_str = f"{bytes_size / 1024:.1f} KB"

        cam_status = 'Aktif' if (camera_processor is not None and camera_processor.is_running) else 'Kapalı'
        cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        ram_pct = round(psutil.virtual_memory().percent, 1)

        return jsonify({
            'success': True,
            'python_version': str(sys.version.split()[0]),
            'platform': f"{platform.system()} {platform.release()}",
            'db_size': db_size_str,
            'face_lib': 'YuNet + SFace (Deep Learning)',
            'camera_status': cam_status,
            'last_update': datetime.datetime.now().strftime('%H:%M:%S'),
            'cpu_usage': cpu_pct,
            'ram_usage': ram_pct,
            'opencv_version': str(cv2.__version__),
            'yolo_available': bool(HAS_YOLO),
        })
    except Exception as e:
        logger.error(f"System info error: {e}")
        return jsonify({
            'success': True,
            'python_version': str(sys.version.split()[0]),
            'platform': 'Windows',
            'db_size': 'N/A',
            'face_lib': 'YuNet + SFace',
            'camera_status': 'Kapalı',
            'last_update': datetime.datetime.now().strftime('%H:%M:%S'),
            'cpu_usage': 0,
            'ram_usage': 0,
        })



@app.route('/api/database/cleanup', methods=['POST'])
def api_database_cleanup():
    """Manuel veritabanı temizliği tetikler."""
    if veritabanlarini_temizle is None:
        return jsonify({'success': False, 'message': 'Temizleme modülü yüklü değil.'}), 500

    data = request.get_json() or {}
    merkezi_cfg = config.get('merkezi_db', {})
    local_retention = data.get('local_retention_days', merkezi_cfg.get('local_retention_days', 7))
    pg_retention = data.get('pg_retention_days', merkezi_cfg.get('pg_retention_days', 30))

    try:
        from pg_sync import pg_baglan
        engine = pg_baglan(merkezi_cfg)
        result = veritabanlarini_temizle(
            db_mgr=db_manager,
            engine=engine,
            local_retention_days=int(local_retention),
            pg_retention_days=int(pg_retention)
        )
        msg = f"Temizlik tamamlandı. Yerel SQLite: {result.get('local_deleted', 0)} kayıt, PostgreSQL: {result.get('pg_deleted', 0)} kayıt silindi."
        return jsonify({'success': True, 'message': msg, 'details': result})
    except Exception as e:
        logger.error(f"Manuel veritabanı temizleme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera/list', methods=['GET'])
@app.route('/api/cameras/scan', methods=['GET'])
def api_scan_cameras():
    cam_list = scan_cameras()
    return jsonify({'cameras': cam_list, 'camera_list': cam_list})


UPLOAD_VIDEO_DIR = BASE_DIR / 'web' / 'static' / 'uploads' / 'videos'
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

@app.route('/api/video/upload', methods=['POST'])
def api_upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'Video dosyası bulunamadı.'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Dosya seçilmedi.'}), 400
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return jsonify({'success': False, 'error': 'Desteklenmeyen video formatı! (.mp4, .avi, .mov, .mkv, .webm)'}), 400

    filename = f"video_{int(time.time())}_{secure_filename(file.filename)}"
    save_path = UPLOAD_VIDEO_DIR / filename
    file.save(str(save_path))
    
    return jsonify({
        'success': True,
        'message': 'Video başarıyla yüklendi.',
        'video_path': str(save_path.resolve()),
        'filename': filename
    })

@app.route('/api/video/list', methods=['GET'])
def api_list_videos():
    videos = []
    if UPLOAD_VIDEO_DIR.exists():
        for f in UPLOAD_VIDEO_DIR.glob('*'):
            if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                videos.append({
                    'filename': f.name,
                    'path': str(f.resolve()),
                    'size_mb': round(f.stat().st_size / (1024 * 1024), 2)
                })
    return jsonify({'videos': videos})


@app.route('/api/video/delete', methods=['POST', 'DELETE'])
@login_required
def api_delete_video():
    global camera_processor
    data = request.get_json() or {}
    filename = data.get('filename') or data.get('video_path')
    if not filename:
        return jsonify({'success': False, 'error': 'Silinecek video dosya adı belirtilmedi.'}), 400

    clean_filename = Path(filename).name
    target_path = UPLOAD_VIDEO_DIR / clean_filename

    if camera_processor and camera_processor.is_running:
        curr_source = str(getattr(camera_processor, 'camera_id', ''))
        if clean_filename in curr_source:
            camera_processor.stop_camera()

    if target_path.exists():
        try:
            target_path.unlink()
            logger.info(f"Video dosyası silindi: {target_path}")
            return jsonify({'success': True, 'message': 'Video başarıyla silindi.'})
        except Exception as e:
            logger.error(f"Video silme hatası: {e}")
            return jsonify({'success': False, 'error': f'Video silinirken hata oluştu: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'error': 'Video dosyası bulunamadı.'}), 404


@app.route('/api/camera/start', methods=['POST'])
@app.route('/api/cameras/start', methods=['POST'])
@app.route('/api/video/start', methods=['POST'])
def api_start_camera():
    global camera_processor
    data = request.get_json() or {}
    source_type = data.get('source_type', 'camera')
    video_path = data.get('video_path')
    cam_id_raw = data.get('camera_id')

    if source_type == 'video' and video_path:
        target_source = video_path
    else:
        if cam_id_raw is None or cam_id_raw == '':
            target_source = config.get('camera_id', 0)
        else:
            try:
                target_source = int(cam_id_raw)
            except (ValueError, TypeError):
                target_source = str(cam_id_raw)

    patron_id, is_super = get_current_patron_id()
    station_override = None
    if not is_super and patron_id:
        try:
            with db_manager.get_session() as session_check:
                u_check = session_check.get(User, patron_id)
                if u_check and u_check.istasyonlar:
                    stations_list = [s.strip() for s in u_check.istasyonlar.split(',') if s.strip()]
                    if stations_list:
                        station_override = stations_list[0]
        except Exception:
            pass

    cfg = dict(config)
    cfg['camera_id'] = target_source
    if station_override:
        cfg['station_name'] = station_override
        cfg['istasyon_adi'] = station_override

    if camera_processor is None:
        camera_processor = CameraProcessor(
            camera_id=target_source,
            config=cfg,
            db_path=str(DB_PATH),
            face_recognizer=face_recognizer,
            socketio=socketio
        )
    else:
        if camera_processor.is_running:
            camera_processor.stop_camera()

        camera_processor.camera_id = target_source
        camera_processor.cfg.update(cfg)
        camera_processor.config.update(cfg)
        if hasattr(camera_processor, '_update_hostname'):
            camera_processor._update_hostname()

    success = camera_processor.start_camera()
    if success:
        label = "Video Dosyası" if source_type == 'video' else f"Kamera {target_source}"
        return jsonify({'success': True, 'message': f'{label} analizi başlatıldı.', 'camera_id': str(target_source)})
    else:
        return jsonify({'success': False, 'message': 'Kaynak başlatılamadı.'}), 400




@app.route('/api/camera/stop', methods=['POST'])
@app.route('/api/cameras/stop', methods=['POST'])
def api_stop_camera():
    global camera_processor
    if camera_processor is not None:
        camera_processor.stop_camera()
    return jsonify({'success': True, 'message': 'Kamera durduruldu.'})


@app.route('/api/workers', methods=['GET'])
def api_workers_list():
    patron_id, is_super, stations = get_current_patron_access()
    try:
        with db_manager.get_session() as session_orm:
            stmt = select(Worker)
            if patron_id != -99999 and not is_super:
                conds = []
                if patron_id and patron_id > 0:
                    conds.append(Worker.patron_id == patron_id)
                if stations:
                    conds.append(Worker.istasyon_adi.in_(stations))
                if conds:
                    stmt = stmt.where(or_(*conds))
                else:
                    stmt = stmt.where(Worker.patron_id == patron_id)
            workers = session_orm.scalars(stmt).all()
            return jsonify([w.to_dict() for w in workers])
    except Exception as e:
        logger.error(f"Workers list hatası: {e}")
        return jsonify([])


@app.route('/api/workers/register', methods=['POST'])
@app.route('/api/workers', methods=['POST'])
@login_required
def api_workers_add():
    data = request.form if request.form else (request.get_json(silent=True) or {})
    ad = str(data.get('ad', '')).strip()
    soyad = str(data.get('soyad', '')).strip()
    sicil_no = str(data.get('sicil_no', '')).strip()
    departman = str(data.get('departman', '')).strip()
    istasyon_adi = str(data.get('istasyon_adi', '')).strip()
    req_patron_id = data.get('patron_id')

    patron_id, is_super = get_current_patron_id()
    target_patron_id = patron_id
    if is_super and req_patron_id and str(req_patron_id).isdigit():
        target_patron_id = int(req_patron_id)
    else:
        target_patron_id = session.get('user_id')

    if not ad or not soyad:
        return jsonify({'success': False, 'message': 'Ad ve Soyad gereklidir.'}), 400

    if not is_super and patron_id and istasyon_adi:
        try:
            with db_manager.get_session() as session_check:
                u_check = session_check.get(User, patron_id)
                assigned_stations = [s.strip() for s in (u_check.istasyonlar or '').split(',') if s.strip()]
                if istasyon_adi not in assigned_stations:
                    return jsonify({
                        'success': False,
                        'message': f"Yetkiniz dışındaki '{istasyon_adi}' istasyonuna çalışan ekleyemezsiniz. Yalnızca size tanımlı istasyonlara ({', '.join(assigned_stations) if assigned_stations else 'hiçbiri'}) çalışan ekleyebilirsiniz."
                    }), 403
        except Exception:
            pass

    if sicil_no:
        try:
            with db_manager.get_session() as session_check:
                existing_sicil = session_check.scalars(
                    select(Worker).where(Worker.sicil_no == sicil_no, Worker.aktif == 1)
                ).first()
                if existing_sicil:
                    return jsonify({
                        'success': False,
                        'message': f"'{sicil_no}' sicil numarası zaten '{existing_sicil.ad} {existing_sicil.soyad}' isimli çalışana aittir! Lütfen farklı bir sicil numarası giriniz."
                    }), 400
        except Exception:
            pass

    try:
        with db_manager.get_session() as session_orm:
            if istasyon_adi:
                existing_w = session_orm.scalars(
                    select(Worker).where(Worker.istasyon_adi == istasyon_adi, Worker.aktif == 1)
                ).first()
                if existing_w:
                    return jsonify({
                        'success': False,
                        'message': f"'{istasyon_adi}' istasyonuna zaten '{existing_w.ad} {existing_w.soyad}' atanmış! Her istasyona en fazla 1 çalışan atanabilir."
                    }), 400

            w = Worker(
                ad=ad,
                soyad=soyad,
                sicil_no=sicil_no or None,
                departman=departman or None,
                istasyon_adi=istasyon_adi or None,
                patron_id=target_patron_id,
                aktif=1
            )
            session_orm.add(w)
            session_orm.commit()
            return jsonify({'success': True, 'message': f'{ad} {soyad} eklendi ve {istasyon_adi or "İstasyon"} üzerine atandı.', 'worker': w.to_dict()})
    except Exception as e:
        logger.error(f"İşçi ekleme hatası: {e}")
        err_str = str(e)
        if 'sicil_no' in err_str or 'UNIQUE constraint' in err_str:
            msg = f"'{sicil_no}' sicil numarası veya girilen bilgiler başka bir çalışan tarafından kullanılmaktadır."
        else:
            msg = f"İşlem gerçekleştirilemedi: {err_str}"
        return jsonify({'success': False, 'message': msg}), 400


@app.route('/api/workers/<int:worker_id>/delete', methods=['POST', 'DELETE'])
@app.route('/api/workers/<int:worker_id>', methods=['DELETE', 'POST'])
@login_required
def api_workers_delete(worker_id):
    try:
        with db_manager.get_session() as session_orm:
            w = session_orm.get(Worker, worker_id)
            if w:
                w_name = f"{w.ad} {w.soyad}".strip()
                session_orm.delete(w)
                session_orm.execute(delete(DurumKaydi).where(DurumKaydi.worker_adi == w_name))
                session_orm.execute(delete(Alarm).where(Alarm.aciklama.like(f"%{w_name}%")))
                session_orm.commit()
                return jsonify({'success': True, 'message': 'Çalışan silindi.'})
        return jsonify({'success': False, 'message': 'Çalışan bulunamadı veya silme işlemi başarısız.'}), 400
    except Exception as e:
        logger.error(f"Silme hatası: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/workers/<int:worker_id>/update', methods=['POST', 'PUT'])
@app.route('/api/workers/<int:worker_id>', methods=['PUT', 'POST'])
@login_required
def api_workers_update(worker_id):
    data = request.form if request.form else (request.get_json(silent=True) or {})
    ad = str(data.get('ad', '')).strip()
    soyad = str(data.get('soyad', '')).strip()
    sicil_no = str(data.get('sicil_no', '')).strip()
    departman = str(data.get('departman', '')).strip()
    istasyon_adi = str(data.get('istasyon_adi', '')).strip()

    patron_id, is_super = get_current_patron_id()

    if not is_super and patron_id and istasyon_adi:
        try:
            with db_manager.get_session() as session_check:
                u_check = session_check.get(User, patron_id)
                assigned_stations = [s.strip() for s in (u_check.istasyonlar or '').split(',') if s.strip()]
                if istasyon_adi not in assigned_stations:
                    return jsonify({
                        'success': False,
                        'message': f"Yetkiniz dışındaki '{istasyon_adi}' istasyonuna çalışan atayamazsınız. Yalnızca size tanımlı istasyonlara ({', '.join(assigned_stations) if assigned_stations else 'hiçbiri'}) atama yapabilirsiniz."
                    }), 403
        except Exception:
            pass

    if sicil_no:
        try:
            with db_manager.get_session() as session_check:
                existing_sicil = session_check.scalars(
                    select(Worker).where(Worker.sicil_no == sicil_no, Worker.id != worker_id, Worker.aktif == 1)
                ).first()
                if existing_sicil:
                    return jsonify({
                        'success': False,
                        'message': f"'{sicil_no}' sicil numarası zaten '{existing_sicil.ad} {existing_sicil.soyad}' isimli çalışana aittir! Lütfen farklı bir sicil numarası giriniz."
                    }), 400
        except Exception:
            pass

    try:
        with db_manager.get_session() as session_orm:
            w = session_orm.get(Worker, worker_id)
            if not w:
                return jsonify({'success': False, 'message': 'Çalışan bulunamadı.'}), 404

            if istasyon_adi:
                existing_w = session_orm.scalars(
                    select(Worker).where(
                        Worker.istasyon_adi == istasyon_adi,
                        Worker.id != worker_id,
                        Worker.aktif == 1
                    )
                ).first()
                if existing_w:
                    return jsonify({
                        'success': False,
                        'message': f"'{istasyon_adi}' istasyonuna zaten '{existing_w.ad} {existing_w.soyad}' atanmış! Her istasyona en fazla 1 çalışan atanabilir."
                    }), 400

            if ad: w.ad = ad
            if soyad: w.soyad = soyad
            if sicil_no: w.sicil_no = sicil_no
            if departman: w.departman = departman
            if 'istasyon_adi' in data: w.istasyon_adi = istasyon_adi or None
            session_orm.commit()
            return jsonify({'success': True, 'message': 'Çalışan ve atanan istasyon güncellendi.', 'worker': w.to_dict()})
    except Exception as e:
        logger.error(f"Güncelleme hatası: {e}")
        err_str = str(e)
        if 'sicil_no' in err_str or 'UNIQUE constraint' in err_str:
            msg = f"'{sicil_no}' sicil numarası başka bir çalışan tarafından kullanılmaktadır."
        else:
            msg = f"Güncelleme hatası: {err_str}"
        return jsonify({'success': False, 'message': msg}), 400


@app.route('/api/alarms', methods=['GET'])
def api_alarms():
    """Tüm alarmları ORM sorgusuyla getirir."""
    limit = request.args.get('limit', 10, type=int)
    patron_id, is_super = get_current_patron_id()
    try:
        with db_manager.get_session() as session_orm:
            stmt = select(Alarm)
            if patron_id is not None:
                patron_workers = session_orm.scalars(select(Worker).where(Worker.patron_id == patron_id)).all()
                p_names = [f"{w.ad} {w.soyad}".strip() for w in patron_workers]
                if p_names:
                    stmt = stmt.where(or_(*[Alarm.aciklama.like(f"%{name}%") for name in p_names]))
                else:
                    return jsonify([])
            stmt = stmt.order_by(Alarm.id.desc()).limit(limit)
            alarmlar = session_orm.scalars(stmt).all()
            result = [a.to_dict() for a in alarmlar]
            return jsonify(result)
    except Exception as e:
        logger.error(f"Alarm listesi hatası: {e}")
        return jsonify([])


@app.route('/api/alarms/unread_count', methods=['GET'])
def api_alarms_unread_count():
    try:
        with db_manager.get_session() as session_orm:
            count = session_orm.scalar(select(func.count(Alarm.id)).where(Alarm.okundu == 0)) or 0
            return jsonify({'count': count, 'unread_count': count})
    except Exception as e:
        return jsonify({'count': 0, 'unread_count': 0})


@app.route('/api/alarms/mark_read', methods=['POST'])
def api_alarms_mark_read():
    try:
        with db_manager.get_session() as session_orm:
            unread_alarms = session_orm.scalars(select(Alarm).where(Alarm.okundu == 0)).all()
            for a in unread_alarms:
                a.okundu = 1
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ---------------------------------------------------------------------------
# Ayarlar API
# ---------------------------------------------------------------------------

@app.route('/api/settings/save', methods=['POST'])
@admin_required
def api_settings_save():
    try:
        data = request.get_json() or {}
        config.update(data)

        st_name = data.get('station_name') or data.get('istasyon_adi')
        if st_name:
            config['station_name'] = str(st_name).strip()
            config['istasyon_adi'] = str(st_name).strip()

        # Keep ROI settings in sync between root keys and nested roi dict
        rx1 = data.get('roi_x1', config.get('roi_x1'))
        ry1 = data.get('roi_y1', config.get('roi_y1'))
        rx2 = data.get('roi_x2', config.get('roi_x2'))
        ry2 = data.get('roi_y2', config.get('roi_y2'))

        if isinstance(data.get('roi'), dict):
            roi_in = data['roi']
            rx1 = roi_in.get('x1', roi_in.get('x1_oran', rx1))
            ry1 = roi_in.get('y1', roi_in.get('y1_oran', ry1))
            rx2 = roi_in.get('x2', roi_in.get('x2_oran', rx2))
            ry2 = roi_in.get('y2', roi_in.get('y2_oran', ry2))

        if rx1 is not None:
            fl_rx1 = float(rx1)
            config['roi_x1'] = fl_rx1
        if ry1 is not None:
            fl_ry1 = float(ry1)
            config['roi_y1'] = fl_ry1
        if rx2 is not None:
            fl_rx2 = float(rx2)
            config['roi_x2'] = fl_rx2
        if ry2 is not None:
            fl_ry2 = float(ry2)
            config['roi_y2'] = fl_ry2

        config['roi'] = {
            'x1': float(config.get('roi_x1', 0.0)),
            'y1': float(config.get('roi_y1', 0.0)),
            'x2': float(config.get('roi_x2', 1.0)),
            'y2': float(config.get('roi_y2', 1.0)),
            'x1_oran': float(config.get('roi_x1', 0.0)),
            'y1_oran': float(config.get('roi_y1', 0.0)),
            'x2_oran': float(config.get('roi_x2', 1.0)),
            'y2_oran': float(config.get('roi_y2', 1.0))
        }

        # Keep merkezi_db dict in sync with postgres_* config keys
        if 'merkezi_db' not in config or not isinstance(config['merkezi_db'], dict):
            config['merkezi_db'] = {}
        config['merkezi_db'].update({
            'aktif': config.get('postgres_enabled', True),
            'host': config.get('postgres_host', '127.0.0.1'),
            'port': config.get('postgres_port', 5432),
            'dbname': config.get('postgres_db', 'fabrika_takip'),
            'kullanici': config.get('postgres_user', 'postgres'),
            'sifre': config.get('postgres_password', ''),
            'senkron_araligi_sn': config.get('postgres_sync_interval', 60)
        })

        save_config(config)
        if camera_processor is not None:
            camera_processor.update_config(config)
        return jsonify({'success': True, 'message': 'Ayarlar başarıyla kaydedildi.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/settings/test_db', methods=['POST'])
@admin_required
def api_settings_test_db():
    data = request.get_json() or {}
    try:
        from pg_sync import pg_baglan, pg_baglantiyi_kapat
        conn = pg_baglan(data)
        if conn:
            pg_baglantiyi_kapat(conn)
            return jsonify({'success': True, 'message': 'PostgreSQL bağlantısı başarılı.'})
        else:
            return jsonify({'success': False, 'message': 'PostgreSQL bağlantısı kurulamadı.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/settings/theme', methods=['POST'])
def api_settings_theme():
    return jsonify({'success': True})


def _calculate_worker_durations(session_orm, worker_id=None, worker_name='', start_date='', end_date='', istasyon='', save_interval=5, model_cls=DurumKaydi):
    """
    Belirli bir çalışan, istasyon ve tarih aralığı için ham durum kayıtlarını inceleyerek 
    kesintisiz aktif, inaktif ve telefon sürelerini tam olarak hesaplar.
    """
    zaman_str_expr = func.cast(model_cls.zaman, String)
    filters = []
    if start_date:
        filters.append(func.substr(zaman_str_expr, 1, 10) >= start_date)
    if end_date:
        filters.append(func.substr(zaman_str_expr, 1, 10) <= end_date)

    if istasyon:
        filters.append(model_cls.istasyon_adi == istasyon)

    if istasyon and istasyon.startswith('VIDEO:'):
        pass
    elif worker_id and str(worker_id).isdigit() and int(worker_id) > 0:
        filters.append(or_(
            model_cls.worker_id == int(worker_id),
            model_cls.worker_adi == worker_name
        ))
    elif worker_name and worker_name != 'Atanmamış Çalışan' and not worker_name.startswith('Video:'):
        filters.append(model_cls.worker_adi == worker_name)

    stmt = select(model_cls).where(and_(*filters)).order_by(model_cls.zaman.asc())
    kayitlar = session_orm.scalars(stmt).all()

    aktif_sec = 0
    kaynak_sec = 0
    inaktif_sec = 0
    telefon_sec = 0

    num_kayitlar = len(kayitlar)
    for i in range(num_kayitlar):
        k = kayitlar[i]
        st = (k.durum or '').upper()
        is_kaynak = 'KAYNAK' in st
        is_telefon = 'TELEFON' in st
        is_inaktif = 'İNAKTİF' in st or 'INAKTIF' in st or 'NAKT' in st
        is_aktif = st.startswith('AKT')

        if is_telefon:
            cat = 'TELEFON'
        elif is_kaynak:
            cat = 'KAYNAK'
        elif is_inaktif:
            cat = 'INAKTIF'
        elif is_aktif:
            cat = 'AKTIF'
        else:
            cat = 'INAKTIF'

        try:
            if isinstance(k.zaman, datetime.datetime):
                z_dt = k.zaman
            else:
                z_dt = datetime.datetime.strptime(str(k.zaman).replace('T', ' ')[:19], '%Y-%m-%d %H:%M:%S')
        except Exception:
            z_dt = datetime.datetime.now()

        if i < num_kayitlar - 1:
            next_k = kayitlar[i + 1]
            try:
                if isinstance(next_k.zaman, datetime.datetime):
                    next_dt = next_k.zaman
                else:
                    next_dt = datetime.datetime.strptime(str(next_k.zaman).replace('T', ' ')[:19], '%Y-%m-%d %H:%M:%S')
            except Exception:
                next_dt = z_dt

            gap = (next_dt - z_dt).total_seconds()
            dur = int(gap) if 0 < gap <= 15 else save_interval
        else:
            dur = save_interval


        if cat == 'KAYNAK':
            kaynak_sec += dur
        elif cat == 'AKTIF':
            aktif_sec += dur
        elif cat == 'INAKTIF':
            inaktif_sec += dur
        elif cat == 'TELEFON':
            telefon_sec += dur

    toplam_sec = aktif_sec + kaynak_sec + inaktif_sec + telefon_sec

    return {
        'aktif_sec': aktif_sec,
        'kaynak_sec': kaynak_sec,
        'inaktif_sec': inaktif_sec,
        'telefon_sec': telefon_sec,
        'toplam_sec': toplam_sec,
        'kayitlar_count': len(kayitlar)
    }



def _build_orm_filters(start: str, end: str, istasyon: str, only_registered: bool = False, patron_id: int = None, model_cls=DurumKaydi):
    """Tarih, istasyon, patron_id ve atanmış istasyon filtrelerini ORM koşul listesi olarak oluşturur."""
    filters = []
    zaman_str_expr = func.cast(model_cls.zaman, String)
    if start:
        filters.append(func.substr(zaman_str_expr, 1, 10) >= start)
    if end:
        filters.append(func.substr(zaman_str_expr, 1, 10) <= end)
    if istasyon:
        if 'VIDEO:' in istasyon.upper() or istasyon.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            filters.append(model_cls.istasyon_adi.like(f"%{istasyon}%"))
        else:
            filters.append(model_cls.istasyon_adi == istasyon)

    if patron_id is not None:
        stations = []
        patron_worker_ids = []
        try:
            with db_manager.get_session() as local_session:
                u = local_session.get(User, patron_id)
                if u and u.istasyonlar:
                    stations = [s.strip() for s in u.istasyonlar.split(',') if s.strip()]

                cond_w = [Worker.patron_id == patron_id]
                if stations:
                    cond_w.append(Worker.istasyon_adi.in_(stations))

                patron_worker_ids = local_session.scalars(select(Worker.id).where(or_(*cond_w))).all()
        except Exception:
            pass

        patron_conds = []
        if stations:
            patron_conds.append(model_cls.istasyon_adi.in_(stations))
        if patron_worker_ids:
            patron_conds.append(model_cls.worker_id.in_(patron_worker_ids))

        if patron_conds:
            filters.append(or_(*patron_conds))
        else:
            filters.append(model_cls.worker_id == -1)

    return filters


from contextlib import contextmanager

@contextmanager
def _get_reports_db_context():
    """
    Eğer PostgreSQL (merkezi_db) aktif ve erişilebilir ise PostgreSQL ORM Session ve CentralDurumKaydiModel döner.
    Aksi takdirde yerel SQLite db_manager session ve DurumKaydi döner.
    """
    merkezi_cfg = config.get('merkezi_db', {})
    is_pg_active = merkezi_cfg.get('aktif', True) if isinstance(merkezi_cfg, dict) else False

    if is_pg_active and HAS_PG_SYNC:
        try:
            from pg_sync import pg_baglan, CentralDurumKaydiModel, pg_baglantiyi_kapat
            engine = pg_baglan(merkezi_cfg)
            if engine:
                from sqlalchemy.orm import Session
                session = Session(engine)
                try:
                    yield session, CentralDurumKaydiModel
                except Exception as query_exc:
                    logger.warning(f"PostgreSQL sorgu hatası ({query_exc}), yerel SQLite veritabanına geçiliyor.")
                    session.close()
                    pg_baglantiyi_kapat(engine)
                    with db_manager.get_session() as fallback_session:
                        yield fallback_session, DurumKaydi
                    return
                finally:
                    try:
                        session.close()
                        pg_baglantiyi_kapat(engine)
                    except Exception:
                        pass
                return
        except Exception as e:
            logger.warning(f"PostgreSQL rapor bağlantısı başarısız, yerel SQLite'a geçiliyor: {e}")

    with db_manager.get_session() as session:
        yield session, DurumKaydi


def format_duration_tr(seconds: float) -> str:
    """Saniyeyi insan tarafından rahatça anlaşılır saat / dakika / saniye biçimine dönüştürür."""
    sec_int = int(round(seconds))
    if sec_int <= 0:
        return "0 dk"
    
    hours = sec_int // 3600
    remainder = sec_int % 3600
    minutes = remainder // 60
    secs = remainder % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours} sa")
    if minutes > 0:
        parts.append(f"{minutes} dk")
    if secs > 0 and hours == 0:
        parts.append(f"{secs} sn")
    
    return " ".join(parts) if parts else "0 dk"


def _format_date_tr(date_str: str) -> str:
    if not date_str:
        return ''
    try:
        dt = datetime.datetime.strptime(date_str[:10], '%Y-%m-%d')
        return dt.strftime('%d.%m.%Y')
    except Exception:
        return date_str


@app.route('/api/reports/summary', methods=['GET'])
@login_required
def api_reports_summary():
    """Özet rapor istatistiklerini ORM ile hesaplar."""
    start    = request.args.get('start', '')
    end      = request.args.get('end', '')
    istasyon = request.args.get('istasyon', '')

    save_interval = config.get('save_interval', 5)
    patron_id, is_super = get_current_patron_id()

    try:
        with _get_reports_db_context() as (session_orm, model_cls):
            filters = _build_orm_filters(start, end, istasyon, patron_id=patron_id, model_cls=model_cls)

            # Toplam Aktif Kayıt Sayısı
            stmt_aktif = select(func.count(model_cls.id)).where(model_cls.durum.like('AKT%'))
            # Toplam Kaynak Kayıt Sayısı
            stmt_kaynak = select(func.count(model_cls.id)).where(model_cls.durum.like('%KAYNAK%'))
            # Toplam İnaktif Kayıt Sayısı
            stmt_inaktif = select(func.count(model_cls.id)).where(model_cls.durum.like('%NAKT%'))

            if filters:
                stmt_aktif = stmt_aktif.where(and_(*filters))
                stmt_kaynak = stmt_kaynak.where(and_(*filters))
                stmt_inaktif = stmt_inaktif.where(and_(*filters))

            aktif_cnt = session_orm.scalar(stmt_aktif) or 0
            kaynak_cnt = session_orm.scalar(stmt_kaynak) or 0
            inaktif_cnt = session_orm.scalar(stmt_inaktif) or 0

            # Toplam Alarm
            with db_manager.get_session() as local_session:
                stmt_alarm = select(func.count(Alarm.id))
                alarm_filters = []
                alarm_zaman_expr = func.cast(Alarm.zaman, String)
                if start: alarm_filters.append(func.substr(alarm_zaman_expr, 1, 10) >= start)
                if end: alarm_filters.append(func.substr(alarm_zaman_expr, 1, 10) <= end)
                if istasyon: alarm_filters.append(Alarm.istasyon_adi == istasyon)
                if alarm_filters:
                    stmt_alarm = stmt_alarm.where(and_(*alarm_filters))
                toplam_alarm = local_session.scalar(stmt_alarm) or 0
                
                stmt_workers = select(func.count(Worker.id)).where(Worker.aktif == 1)
                if patron_id is not None:
                    stmt_workers = stmt_workers.where(Worker.patron_id == patron_id)
                toplam_calisan = local_session.scalar(stmt_workers) or 0

        aktif_sure_dk = round((aktif_cnt * save_interval) / 60.0, 1)
        kaynak_sure_dk = round((kaynak_cnt * save_interval) / 60.0, 1)
        inaktif_sure_dk = round((inaktif_cnt * save_interval) / 60.0, 1)
        toplam_sure_dk = aktif_sure_dk + kaynak_sure_dk + inaktif_sure_dk
        verimlilik = round(((aktif_sure_dk + kaynak_sure_dk) / toplam_sure_dk * 100), 1) if toplam_sure_dk > 0 else 0.0

        return jsonify({
            'toplam_calisan': toplam_calisan,
            'aktif_sure_dk': aktif_sure_dk,
            'kaynak_sure_dk': kaynak_sure_dk,
            'inaktif_sure_dk': inaktif_sure_dk,
            'verimlilik_orani': verimlilik,
            'toplam_alarm': toplam_alarm,
            'aktif_kayit': aktif_cnt,
            'kaynak_kayit': kaynak_cnt,
            'inaktif_kayit': inaktif_cnt,
            'aktif_alarm': toplam_alarm,
            'aktif_oran': verimlilik,
            'alarm_count': toplam_alarm,
        })
    except Exception as e:
        logger.error(f"Özet rapor hatası (ORM): {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/chart_data', methods=['GET'])
@app.route('/api/reports/hourly', methods=['GET'])
@login_required
def api_reports_hourly():
    """Saatlik aktif/inaktif dağılımını 24 saatlik eksiksiz etiketlerle ORM ile döndürür."""
    start    = request.args.get('start', request.args.get('date', datetime.date.today().isoformat()))
    end      = request.args.get('end', '')
    istasyon = request.args.get('istasyon', '')
    patron_id, is_super = get_current_patron_id()

    # 24 saatlik varsayılan veri sözlüğü (00:00 .. 23:00)
    hourly_map = {f"{h:02d}:00": {"aktif": 0, "inaktif": 0} for h in range(24)}

    try:
        with _get_reports_db_context() as (session_orm, model_cls):
            filters = _build_orm_filters(start, end, istasyon, patron_id=patron_id, model_cls=model_cls)
            zaman_str_expr = func.cast(model_cls.zaman, String)
            saat_col = func.substr(zaman_str_expr, 12, 2)

            stmt = select(
                saat_col.label('saat'),
                func.sum(case((model_cls.durum.like('AKT%'), 1), else_=0)).label('aktif'),
                func.sum(case((model_cls.durum.like('%NAKT%'), 1), else_=0)).label('inaktif')
            )
            if filters:
                stmt = stmt.where(and_(*filters))

            stmt = stmt.group_by(saat_col).order_by(saat_col)
            rows = session_orm.execute(stmt).all()

            save_interval = config.get('save_interval', 5)
            for r in rows:
                if r.saat and str(r.saat).isdigit():
                    hour_key = f"{int(r.saat):02d}:00"
                    if hour_key in hourly_map:
                        hourly_map[hour_key]["aktif"] = round(((r.aktif or 0) * save_interval) / 60.0, 1)
                        hourly_map[hour_key]["inaktif"] = round(((r.inaktif or 0) * save_interval) / 60.0, 1)

        labels  = list(hourly_map.keys())
        aktif   = [hourly_map[k]["aktif"] for k in labels]
        inaktif = [hourly_map[k]["inaktif"] for k in labels]

        return jsonify({'labels': labels, 'aktif': aktif, 'inaktif': inaktif})
    except Exception as e:
        logger.error(f"Grafik verisi hatası (ORM): {e}")
        labels = [f"{h:02d}:00" for h in range(24)]
        return jsonify({'labels': labels, 'aktif': [0]*24, 'inaktif': [0]*24, 'error': str(e)})


@app.route('/api/reports/data', methods=['GET'])
@app.route('/api/reports/worker_stats', methods=['GET'])
@login_required
def api_reports_worker_stats():
    """Çalışanların günlük bazda çalışma süreleri ve detaylarını ORM ile döndürür."""
    start    = request.args.get('start', '')
    end      = request.args.get('end', '')
    istasyon = request.args.get('istasyon', '')
    worker   = request.args.get('worker', '')

    save_interval = config.get('save_interval', 5)
    patron_id, is_super = get_current_patron_id()

    try:
        with _get_reports_db_context() as (session_orm, model_cls):
            filters = _build_orm_filters(start, end, istasyon, patron_id=patron_id, model_cls=model_cls)
            if worker:
                filters.append(or_(
                    model_cls.worker_id == int(worker) if str(worker).isdigit() else False,
                    model_cls.worker_adi.like(f"%{worker}%")
                ))

            default_st = config.get('station_name') or config.get('istasyon_adi') or 'Istasyon-1'
            if not default_st or default_st in ['auto', 'auto (Otomatik Bilgisayar Adı)'] or default_st.startswith('LAPTOP-') or default_st.startswith('DESKTOP-'):
                default_st = 'Istasyon-1'

            station_worker_map = {}
            try:
                with db_manager.get_session() as local_sess:
                    w_all = local_sess.scalars(select(Worker).where(Worker.aktif == 1)).all()
                    for w in w_all:
                        if w.istasyon_adi and w.istasyon_adi.strip():
                            station_worker_map[w.istasyon_adi.strip()] = (w.id, f"{w.ad} {w.soyad}".strip())
            except Exception:
                pass

            zaman_str_expr = func.cast(model_cls.zaman, String)
            tarih_col = func.substr(zaman_str_expr, 1, 10)
            istasyon_col = func.coalesce(model_cls.istasyon_adi, default_st)

            stmt = select(
                tarih_col.label('tarih'),
                istasyon_col.label('istasyon_adi'),
                func.count(model_cls.id).label('toplam_kayit'),
                func.sum(case((model_cls.durum.like('AKT%'), 1), else_=0)).label('aktif_kayit'),
                func.sum(case((model_cls.durum.like('%KAYNAK%'), 1), else_=0)).label('kaynak_kayit'),
                func.sum(case((model_cls.durum.like('%NAKT%'), 1), else_=0)).label('inaktif_kayit'),
                func.sum(case((model_cls.durum.like('%TELEFON%'), 1), else_=0)).label('telefon_kayit'),
                func.min(model_cls.zaman).label('ilk_gorulme'),
                func.max(model_cls.zaman).label('son_gorulme')
            )
            if filters:
                stmt = stmt.where(and_(*filters))

            stmt = stmt.group_by(tarih_col, istasyon_col).order_by(desc('tarih'), desc('toplam_kayit'))
            rows = session_orm.execute(stmt).all()

            workers_data = []
            for r in rows:
                st_name = r.istasyon_adi
                if not st_name or st_name.lower() == 'auto' or st_name.startswith('LAPTOP-') or st_name.startswith('DESKTOP-'):
                    st_name = default_st

                w_tuple = station_worker_map.get(st_name)
                w_id = w_tuple[0] if w_tuple else None
                if st_name and st_name.startswith('VIDEO:'):
                    clean_vid_name = st_name.replace('VIDEO: ', '').strip()
                    w_name = f"Video: {clean_vid_name}"
                else:
                    w_name = w_tuple[1] if w_tuple else 'Atanmamış Çalışan'

                tarih_val = r.tarih or ''
                toplam = r.toplam_kayit or 0

                dur_info = _calculate_worker_durations(
                    session_orm,
                    worker_id=w_id,
                    worker_name=w_name if w_id else '',
                    start_date=tarih_val,
                    end_date=tarih_val,
                    istasyon=st_name,
                    save_interval=save_interval,
                    model_cls=model_cls
                )
                
                aktif_sec = dur_info['aktif_sec']
                kaynak_sec = dur_info.get('kaynak_sec', 0)
                inaktif_sec = dur_info['inaktif_sec']
                telefon_sec = dur_info['telefon_sec']
                toplam_sec = dur_info['toplam_sec'] or (toplam * save_interval)

                aktif_min = round(aktif_sec / 60.0, 1)
                kaynak_min = round(kaynak_sec / 60.0, 1)
                inaktif_min = round(inaktif_sec / 60.0, 1)

                uretim_sec = aktif_sec + kaynak_sec
                rate = round((uretim_sec / toplam_sec * 100), 1) if toplam_sec > 0 else 0.0

                ilk_raw = str(r.ilk_gorulme) if r.ilk_gorulme else ''
                son_raw = str(r.son_gorulme) if r.son_gorulme else ''
                ilk_str = ilk_raw.replace('T', ' ')[:19] if ilk_raw else '—'
                son_str = son_raw.replace('T', ' ')[:19] if son_raw else '—'

                aktif_fmt = format_duration_tr(aktif_sec)
                kaynak_fmt = format_duration_tr(kaynak_sec)
                inaktif_fmt = format_duration_tr(inaktif_sec)
                telefon_fmt = format_duration_tr(telefon_sec)

                workers_data.append({
                    'tarih': tarih_val,
                    'tarih_fmt': _format_date_tr(tarih_val),
                    'istasyon_adi': st_name,
                    'worker_id': w_id,
                    'worker_adi': w_name,
                    'toplam_kayit': toplam,
                    'toplam_sure_sec': toplam_sec,
                    'toplam_sure_min': round(toplam_sec / 60.0, 1),
                    'aktif_kayit': r.aktif_kayit or 0,
                    'aktif_sure_sec': aktif_sec,
                    'aktif_sure_min': aktif_min,
                    'aktif_sure_fmt': aktif_fmt,
                    'aktif_saat': round(aktif_min / 60.0, 2),
                    'kaynak_kayit': getattr(r, 'kaynak_kayit', 0) or 0,
                    'kaynak_sure_sec': kaynak_sec,
                    'kaynak_sure_min': kaynak_min,
                    'kaynak_sure_fmt': kaynak_fmt,
                    'kaynak_saat': round(kaynak_min / 60.0, 2),
                    'inaktif_kayit': r.inaktif_kayit or 0,
                    'inaktif_sure_sec': inaktif_sec,
                    'inaktif_sure_min': inaktif_min,
                    'inaktif_sure_fmt': inaktif_fmt,
                    'inaktif_saat': round(inaktif_min / 60.0, 2),
                    'telefon_sure_sec': telefon_sec,
                    'telefon_sure_fmt': telefon_fmt,
                    'verimlilik_orani': rate,
                    'aktif_oran': rate,
                    'ilk_gorulme': ilk_str,
                    'son_gorulme': son_str,
                })

        return jsonify({'workers': workers_data, 'data': workers_data})
    except Exception as e:
        logger.error(f"Çalışan rapor hatası (ORM): {e}")
        return jsonify({'workers': [], 'data': [], 'error': str(e)})


@app.route('/reports/worker_detail_page', methods=['GET'])
@app.route('/worker_analysis', methods=['GET'])
@login_required
def worker_detail_page():
    return render_template('worker_detail_page.html')


@app.route('/api/reports/worker_detail', methods=['GET'])
@login_required
def api_reports_worker_detail():
    """Belirli bir çalışanın detaylı zaman, grafik ve durum kayıtlarını ORM ile döndürür."""
    worker_id   = request.args.get('worker_id')
    worker_name = request.args.get('worker_name', '')
    istasyon    = request.args.get('istasyon', '')
    start       = request.args.get('start', '')
    end         = request.args.get('end', '')

    save_interval = config.get('save_interval', 5)

    try:
        with _get_reports_db_context() as (session_orm, model_cls):
            # 1. Worker Bilgilerini Oku
            worker_obj = None
            with db_manager.get_session() as local_session:
                if worker_id and str(worker_id).isdigit() and int(worker_id) > 0:
                    worker_obj = local_session.get(Worker, int(worker_id))
                if not worker_obj and worker_name:
                    stmt_w = select(Worker).where(or_(
                        Worker.ad.like(f"%{worker_name}%"),
                        Worker.soyad.like(f"%{worker_name}%")
                    ))
                    worker_obj = local_session.scalars(stmt_w).first()
                if not worker_obj and istasyon:
                    stmt_w = select(Worker).where(Worker.istasyon_adi == istasyon, Worker.aktif == 1)
                    worker_obj = local_session.scalars(stmt_w).first()

            patron_id, is_super = get_current_patron_id()
            if not is_super and patron_id:
                has_access = False
                with db_manager.get_session() as local_session:
                    u = local_session.get(User, patron_id)
                    assigned_stations = [s.strip() for s in (u.istasyonlar or '').split(',') if s.strip()]
                    if worker_obj and worker_obj.patron_id == patron_id:
                        has_access = True
                    elif istasyon and istasyon in assigned_stations:
                        has_access = True
                    elif worker_obj and worker_obj.istasyon_adi in assigned_stations:
                        has_access = True

                if not has_access:
                    return jsonify({'success': False, 'message': 'Bu çalışanın raporlarını görüntüleme yetkiniz yoktur.'}), 403

            full_name = f"{worker_obj.ad} {worker_obj.soyad}".strip() if worker_obj else (worker_name or 'Bilinmeyen Çalışan')
            sicil_no  = worker_obj.sicil_no if worker_obj and worker_obj.sicil_no else 'EMP-001'
            departman = worker_obj.departman if worker_obj and worker_obj.departman else 'Üretim'
            photo_url = f"/static/workers/{worker_obj.id}.jpg" if worker_obj else None

            w_id_target = worker_obj.id if worker_obj else worker_id

            if not istasyon and worker_obj and worker_obj.istasyon_adi:
                istasyon = worker_obj.istasyon_adi

            # ── Filtreler: worker_stats ile AYNI mantık ─────────────────────
            # _build_orm_filters tarih + istasyon + patron kısıtlarını uygular
            filters_k = _build_orm_filters(start, end, istasyon=istasyon, patron_id=patron_id, model_cls=model_cls)

            # Çalışana özgü filtre: ID öncelikli, yoksa isimle eşleştir (Video raporlarında istasyon filtresi yeterlidir)
            if istasyon and istasyon.startswith('VIDEO:'):
                pass
            elif w_id_target and str(w_id_target).isdigit() and int(w_id_target) > 0:
                filters_k.append(or_(
                    model_cls.worker_id == int(w_id_target),
                    model_cls.worker_adi == full_name
                ))
            elif full_name and full_name != 'Atanmamış Çalışan' and not full_name.startswith('Video:'):
                filters_k.append(model_cls.worker_adi == full_name)

            stmt_k = select(model_cls).where(and_(*filters_k)).order_by(model_cls.zaman.asc())
            kayitlar = session_orm.scalars(stmt_k).all()

            # 2. Süre Hesabı: worker_stats ile AYNI _calculate_worker_durations çağrısı
            dur_info = _calculate_worker_durations(
                session_orm,
                worker_id=w_id_target,
                worker_name=full_name,
                start_date=start,
                end_date=end,
                istasyon=istasyon,
                save_interval=save_interval,
                model_cls=model_cls
            )

        aktif_sec = dur_info['aktif_sec']
        kaynak_sec = dur_info.get('kaynak_sec', 0)
        inaktif_sec = dur_info['inaktif_sec']
        telefon_sec = dur_info['telefon_sec']
        toplam_sec = dur_info['toplam_sec'] or (aktif_sec + kaynak_sec + inaktif_sec + telefon_sec)

        telefon_cnt = sum(1 for k in kayitlar if 'TELEFON' in (k.durum or '').upper())
        kaynak_cnt = sum(1 for k in kayitlar if 'KAYNAK' in (k.durum or '').upper())

        # 3. Ardışık Kayıtları Zaman Bloklarına Dönüştür (Kaydedildi yerine gerçek süre yazılır)
        recent_records = []
        if kayitlar:
            current_block = None
            for i, k in enumerate(kayitlar):
                z_str = str(k.zaman).replace('T', ' ')[:19]
                dur_lbl = k.durum or 'Bilinmiyor'
                ist_lbl = k.istasyon_adi or 'İstasyon 1'

                try:
                    if isinstance(k.zaman, datetime.datetime):
                        z_dt = k.zaman
                    else:
                        z_dt = datetime.datetime.strptime(z_str, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    z_dt = datetime.datetime.now()

                is_gap_too_large = False
                if i < len(kayitlar) - 1:
                    next_k = kayitlar[i + 1]
                    try:
                        if isinstance(next_k.zaman, datetime.datetime):
                            next_dt = next_k.zaman
                        else:
                            next_dt = datetime.datetime.strptime(str(next_k.zaman).replace('T', ' ')[:19], '%Y-%m-%d %H:%M:%S')
                        gap = int((next_dt - z_dt).total_seconds())
                        is_gap_too_large = (gap > 20)
                        dur_sec = save_interval if is_gap_too_large else (gap if (0 < gap <= 20) else save_interval)
                    except Exception:
                        dur_sec = save_interval
                else:
                    dur_sec = save_interval

                if current_block is None:
                    current_block = {
                        'start_str': z_str,
                        'end_str': z_str,
                        'durum': dur_lbl,
                        'istasyon_adi': ist_lbl,
                        'sure_sec': dur_sec
                    }
                elif current_block['durum'] == dur_lbl and current_block['istasyon_adi'] == ist_lbl and not is_gap_too_large:
                    current_block['end_str'] = z_str
                    current_block['sure_sec'] += dur_sec
                else:
                    t1 = current_block['start_str'][11:16] if len(current_block['start_str']) >= 16 else current_block['start_str']
                    t2 = current_block['end_str'][11:16] if len(current_block['end_str']) >= 16 else current_block['end_str']
                    range_str = f"{t1} - {t2}" if t1 != t2 else t1
                    recent_records.append({
                        'zaman_araligi': range_str,
                        'baslangic': current_block['start_str'],
                        'bitis': current_block['end_str'],
                        'sure_sec': current_block['sure_sec'],
                        'sure_fmt': format_duration_tr(current_block['sure_sec']),
                        'istasyon_adi': current_block['istasyon_adi'],
                        'durum': current_block['durum']
                    })
                    current_block = {
                        'start_str': z_str,
                        'end_str': z_str,
                        'durum': dur_lbl,
                        'istasyon_adi': ist_lbl,
                        'sure_sec': dur_sec
                    }


            if current_block:
                t1 = current_block['start_str'][11:16] if len(current_block['start_str']) >= 16 else current_block['start_str']
                t2 = current_block['end_str'][11:16] if len(current_block['end_str']) >= 16 else current_block['end_str']
                range_str = f"{t1} - {t2}" if t1 != t2 else t1
                recent_records.append({
                    'zaman_araligi': range_str,
                    'baslangic': current_block['start_str'],
                    'bitis': current_block['end_str'],
                    'sure_sec': current_block['sure_sec'],
                    'sure_fmt': format_duration_tr(current_block['sure_sec']),
                    'istasyon_adi': current_block['istasyon_adi'],
                    'durum': current_block['durum']
                })
            recent_records = list(reversed(recent_records))[:100]

        # 4. Saatlik Dağılım Hesabı (Bar/Line Chart için 24 saatlik veriler)
        hourly_aktif = [0] * 24
        hourly_kaynak = [0] * 24
        hourly_inaktif = [0] * 24
        hourly_telefon = [0] * 24

        for k in kayitlar:
            try:
                if isinstance(k.zaman, datetime.datetime):
                    h = k.zaman.hour
                else:
                    h = int(str(k.zaman)[11:13])
                st = (k.durum or '').upper()
                if 'TELEFON' in st:
                    hourly_telefon[h] += save_interval
                elif 'KAYNAK' in st:
                    hourly_kaynak[h] += save_interval
                elif 'İNAKTİF' in st or 'INAKTIF' in st or 'NAKT' in st:
                    hourly_inaktif[h] += save_interval
                elif st.startswith('AKT'):
                    hourly_aktif[h] += save_interval
            except Exception:
                pass

        total_calc = max(toplam_sec, 1)
        aktif_pct = round((aktif_sec / total_calc * 100), 1)
        kaynak_pct = round((kaynak_sec / total_calc * 100), 1)
        inaktif_pct = round((inaktif_sec / total_calc * 100), 1)
        telefon_pct = round((telefon_sec / total_calc * 100), 1)

        uretim_sec = aktif_sec + kaynak_sec
        verimlilik_orani = round((uretim_sec / total_calc * 100), 1) if total_calc > 0 else 0.0

        return jsonify({
            'worker_name': full_name,
            'sicil_no': sicil_no,
            'departman': departman,
            'istasyon_adi': istasyon or (worker_obj.istasyon_adi if worker_obj and worker_obj.istasyon_adi else '') or config.get('station_name', 'İstasyon-1'),
            'photo_url': photo_url,
            'aktif_fmt': format_duration_tr(aktif_sec),
            'aktif_pct': aktif_pct,
            'aktif_sec': aktif_sec,
            'kaynak_fmt': format_duration_tr(kaynak_sec),
            'kaynak_count': kaynak_cnt,
            'kaynak_pct': kaynak_pct,
            'kaynak_sec': kaynak_sec,
            'inaktif_fmt': format_duration_tr(inaktif_sec),
            'inaktif_pct': inaktif_pct,
            'inaktif_sec': inaktif_sec,
            'telefon_fmt': format_duration_tr(telefon_sec),
            'telefon_count': telefon_cnt,
            'telefon_sec': telefon_sec,
            'telefon_pct': telefon_pct,
            'verimlilik_orani': verimlilik_orani,
            'recent_records': recent_records,
            'hourly_labels': [f"{h:02d}:00" for h in range(24)],
            'hourly_aktif_min': [round(s / 60.0, 1) for s in hourly_aktif],
            'hourly_kaynak_min': [round(s / 60.0, 1) for s in hourly_kaynak],
            'hourly_inaktif_min': [round(s / 60.0, 1) for s in hourly_inaktif],
            'hourly_telefon_min': [round(s / 60.0, 1) for s in hourly_telefon]
        })
    except Exception as e:
        logger.error(f"Çalışan detay hatası (ORM): {e}")
        return jsonify({'error': str(e)})



@app.route('/api/cameras/stations', methods=['GET'])
def api_camera_stations():
    try:
        stations_set = {'Istasyon-1', 'Istasyon-2', 'Istasyon-3', 'Istasyon-4'}
        with db_manager.get_session() as session_orm:
            stmt = select(DurumKaydi.istasyon_adi).where(DurumKaydi.istasyon_adi.isnot(None)).distinct()
            db_stations = session_orm.scalars(stmt).all()
            for s in db_stations:
                if s and not s.startswith('VIDEO:') and not s.startswith('LAPTOP-') and not s.startswith('DESKTOP-'):
                    stations_set.add(s)
            
            w_stations = session_orm.scalars(select(Worker.istasyon_adi).where(Worker.istasyon_adi.isnot(None)).distinct()).all()
            for s in w_stations:
                if s and not s.startswith('VIDEO:'):
                    stations_set.add(s)

        sorted_list = sorted(list(stations_set))
        return jsonify({'stations': sorted_list, 'success': True})
    except Exception as e:
        return jsonify({'stations': ['Istasyon-1', 'Istasyon-2', 'Istasyon-3', 'Istasyon-4'], 'success': False})

# ---------------------------------------------------------------------------
# Başlatma
# ---------------------------------------------------------------------------

def _print_banner():
    try:
        banner = """
==========================================================
          ISCI TAKIP SISTEMI - Web Arayuzu (ORM)

  Tarayicida acin: http://localhost:5000
  Yerel ag:        http://0.0.0.0:5000

  Durdurmak icin: Ctrl+C
==========================================================
"""
        print(banner)
    except Exception:
        pass


def initialize():
    """Uygulama başlangıç işlemleri."""
    global face_recognizer, config, camera_processor

    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / 'static').mkdir(parents=True, exist_ok=True)
    (WEB_DIR / 'templates').mkdir(parents=True, exist_ok=True)

    config = load_config()
    logger.info("Yapılandırma yüklendi.")

    init_db()
    face_recognizer = None

    broadcast_thread = threading.Thread(
        target=_broadcast_status,
        name='BroadcastThread',
        daemon=True,
    )
    broadcast_thread.start()
    logger.info("Durum yayın iş parçacığı başlatıldı.")

    merkezi_db_cfg = config.get("merkezi_db") or config
    if HAS_PG_SYNC and SenkronThread:
        try:
            istasyon_adi = config.get("station_name") or config.get("istasyon_adi") or "Istasyon-1"
            if not istasyon_adi or str(istasyon_adi).strip().lower() == "auto":
                istasyon_adi = "Istasyon-1"
            senkron_thread = SenkronThread(
                db_mgr=db_manager,
                merkezi_db_cfg=merkezi_db_cfg,
                istasyon_adi=istasyon_adi,
            )
            senkron_thread.start()
            logger.info("Otomatik PostgreSQL senkronizasyon thread'i başlatıldı.")
        except Exception as e:
            logger.error(f"PostgreSQL senkronizasyon başlatılamadı: {e}")

    # Otomatik Kamera Başlatma (Sistem açıldığında kamerayı beklemeden başlatır)
    if config.get("auto_start_camera", True):
        try:
            target_source = config.get('camera_id', 0)
            cfg = dict(config)
            camera_processor = CameraProcessor(
                camera_id=target_source,
                config=cfg,
                db_path=str(DB_PATH),
                face_recognizer=face_recognizer,
                socketio=socketio
            )
            if camera_processor.start_camera():
                logger.info(f"Kamera (ID: {target_source}) sistem açılışında otomatik olarak başlatıldı.")
            else:
                logger.warning(f"Kamera (ID: {target_source}) sistem açılışında otomatik başlatılamadı.")
        except Exception as e:
            logger.error(f"Otomatik kamera başlatma hatası: {e}")

# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    initialize()
    _print_banner()
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )