"""
core/camera_manager.py
İşçi Takip Sistemi - Kamera İşleme Modülü (Tam ORM Yapısı)

YOLO tabanlı poz tahmini + telefon tespiti + yüz tanıma entegrasyonu.
Tüm veritabanı kayıtları SQLAlchemy 2.0 Code-First ORM ile yönetilir.
"""

import os
import sys
import time
import socket
import logging
import datetime
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
from sqlalchemy import select

from core.database.models import DurumKaydi, Alarm, GunlukOzet, Worker
from core.database.connection import DatabaseManager, db_manager

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None

logger = logging.getLogger(__name__)

# Durum sabitleri
DURUM_AKTIF        = 'AKTİF (Çalışıyor)'
DURUM_KAYNAK       = 'Kaynak Yapıyor'
DURUM_TOLERANS     = 'AKTİF (Üretim Toleransı)'
DURUM_INAKTIF      = 'İNAKTİF (Hareketsiz / Alan Dışı)'
DURUM_TELEFON      = 'İNAKTİF (Telefon Kullanımı!)'
DURUM_TESPIT_YOK   = 'İşçi Tespit Edilemedi'


COCO_CELL_PHONE = 67

SKELETON_CONNECTIONS = [
    (5, 6),                              # Omuzlar
    (5, 7), (7, 9),                      # Sol Kol
    (6, 8), (8, 10),                     # Sağ Kol
    (5, 11), (6, 12), (11, 12),          # Gövde
    (11, 13), (13, 15),                  # Sol Bacak
    (12, 14), (14, 16)                   # Sağ Bacak
]



def tr_to_ascii(text: str) -> str:
    """OpenCV cv2.putText için Türkçe karakterleri ASCII karşılıklarına dönüştürür."""
    if not text:
        return ""
    mapping = {
        'İ': 'I', 'I': 'I', 'ı': 'i',
        'Ğ': 'G', 'ğ': 'g',
        'Ü': 'U', 'ü': 'u',
        'Ş': 'S', 'ş': 's',
        'Ö': 'O', 'ö': 'o',
        'Ç': 'C', 'ç': 'c',
    }
    for tr_char, ascii_char in mapping.items():
        text = text.replace(tr_char, ascii_char)
    return text


class CameraProcessor:
    def __init__(
        self,
        camera_id=0,
        config: dict = None,
        db_path: str = None,
        face_recognizer=None,
        socketio=None,
    ):
        self.camera_id = camera_id
        self.cfg = config or {}
        self.config = self.cfg
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path) if db_path else db_manager
        self.face_recognizer = face_recognizer
        self.socketio = socketio

        self.running = False
        self.is_running = False
        self.cap = None

        self.is_headless = (sys.platform != 'win32' and not os.environ.get('DISPLAY')) or bool(self.cfg.get('headless_mode', False))

        self._frame_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None

        self._update_hostname()

        self._status: Dict[str, Any] = {
            'durum': DURUM_TESPIT_YOK,
            'renk': '#888888',
            'fps': 0.0,
            'kisi_sayisi': 0,
            'istasyon': self._hostname,
            'zaman': datetime.datetime.now().isoformat(),
            'running': False,
            'worker_name': '',
            'worker_confidence': 0.0,
            'phone_detected': False,
            'camera_id': str(camera_id),
        }
        self.current_status = self._status

        self._last_saved_durum = None
        self._last_saved_worker_id = None
        self._last_event_save_time = 0.0

        self._last_station_worker_id = None
        self._last_station_worker_name = ''
        self._last_station_worker_time = 0.0
        self._last_worker_seen_time = 0.0



        self._pose_model = None
        self._det_model = None
        self._welding_model = None
        self.pose_model = None
        self.det_model = None
        self.welding_model = None


    def _update_hostname(self):
        if isinstance(self.camera_id, str) and not str(self.camera_id).isdigit() and ('.' in str(self.camera_id) or '/' in str(self.camera_id) or '\\' in str(self.camera_id)):
            self._hostname = f"VIDEO: {Path(self.camera_id).name}"
        else:
            st_name = self.cfg.get("station_name") or self.cfg.get("istasyon_adi")
            if not st_name or str(st_name).strip().lower() == "auto":
                st_name = "Istasyon-1"
            self._hostname = str(st_name).strip()
        self.station_name = self._hostname

    def _get_station_worker(self):
        """Veritabanından bu istasyona (self._hostname) atanmış aktif çalışanı sorgular."""
        now_t = time.time()
        if hasattr(self, '_last_worker_check_time') and (now_t - self._last_worker_check_time < 10.0):
            return getattr(self, '_cached_assigned_worker', (None, f"{self._hostname} Çalışanı"))

        self._last_worker_check_time = now_t
        w_id = None
        w_name = f"{self._hostname} Çalışanı"
        try:
            with self.db_manager.get_session() as session:
                stmt = select(Worker).where(
                    Worker.istasyon_adi == self._hostname,
                    Worker.aktif == 1
                )
                w = session.scalars(stmt).first()
                if w:
                    w_id = w.id
                    w_name = f"{w.ad} {w.soyad}"
        except Exception as e:
            logger.debug(f"İstasyon çalışan sorgu hatası: {e}")

        self._cached_assigned_worker = (w_id, w_name)
        return w_id, w_name


    def update_config(self, new_config: dict):
        """Web arayüzünden kaydedilen yeni ROI ve ayarları canlı olarak günceller."""
        if not new_config:
            return
        self.cfg.update(new_config)
        self.config.update(new_config)
        self._update_hostname()
        self._update_status({'istasyon': self._hostname, 'station': self._hostname})
        logger.info(f"Kamera {self.camera_id} konfigürasyonu canlı güncellendi. İstasyon: {self._hostname}")

    def _load_models(self):
        if self._pose_model is not None and self._det_model is not None:
            return True

        if not HAS_YOLO:
            logger.warning("ultralytics paketi kurulu değil! YOLO modelleri çalışmayacak.")
            return False

        try:
            import torch
            try:
                torch.set_num_threads(min(4, os.cpu_count() or 4))
            except Exception:
                pass

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                logger.info("NVIDIA CUDA GPU donanım hızlandırması aktif edildi (device=cuda).")
            else:
                logger.info("CUDA GPU bulunamadı. YOLO modelleri PyTorch multi-thread CPU modunda çalışıyor.")

            base_dir = Path(__file__).parent.parent
            pose_path    = base_dir / self.cfg.get('pose_model_path', 'yolov8n-pose.pt')
            det_path     = base_dir / self.cfg.get('det_model_path', 'yolov8n.pt')
            welding_path = base_dir / self.cfg.get('welding_model_path', 'welding_det.pt')

            dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)

            if pose_path.exists():
                logger.info(f"Poz modeli yükleniyor: {pose_path}")
                self._pose_model = YOLO(str(pose_path))
                if use_cuda:
                    try:
                        self._pose_model.to('cuda')
                    except Exception:
                        pass
                try:
                    self._pose_model(dummy_frame, imgsz=320, verbose=False)
                except Exception:
                    pass
                self.pose_model = self._pose_model

            if det_path.exists():
                logger.info(f"Tespit modeli yükleniyor: {det_path}")
                self._det_model = YOLO(str(det_path))
                if use_cuda:
                    try:
                        self._det_model.to('cuda')
                    except Exception:
                        pass
                try:
                    self._det_model(dummy_frame, conf=0.30, imgsz=320, verbose=False)
                except Exception:
                    pass
                self.det_model = self._det_model

            if welding_path.exists():
                logger.info(f"Kaynak tespit modeli yükleniyor: {welding_path}")
                self._welding_model = YOLO(str(welding_path))
                if use_cuda:
                    try:
                        self._welding_model.to('cuda')
                    except Exception:
                        pass
                try:
                    self._welding_model(dummy_frame, conf=0.30, imgsz=320, verbose=False)
                except Exception:
                    pass
                self.welding_model = self._welding_model

            logger.info("YOLO modelleri donanım hızlandırmalı ve ön-ısıtmalı (warmup) olarak yüklendi.")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            self._pose_model = None
            self._det_model = None
            self._welding_model = None
            return False

    def _open_camera(self):
        import platform

        cam_id = self.camera_id
        system = platform.system()

        # Sistem açılışında kamera donanımının hazır olması için otomatik deneme döngüsü (5 deneme)
        max_attempts = 5 if not isinstance(cam_id, str) or str(cam_id).isdigit() else 1
        cap = None

        for attempt in range(max_attempts):
            if isinstance(cam_id, int) or (isinstance(cam_id, str) and str(cam_id).isdigit()):
                cam_idx = int(cam_id)
                if system == 'Windows':
                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                    if not (cap and cap.isOpened()):
                        if cap: cap.release()
                        cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
                    if not (cap and cap.isOpened()):
                        if cap: cap.release()
                        cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
                else:
                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
                    if not (cap and cap.isOpened()):
                        if cap: cap.release()
                        cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
            else:
                cap = cv2.VideoCapture(str(cam_id))

            if cap and cap.isOpened():
                break

            if cap:
                cap.release()
                cap = None

            if attempt < max_attempts - 1:
                logger.info(f"Kamera ({cam_id}) bekleniyor ve açılmaya çalışılıyor... (Deneme {attempt + 1}/{max_attempts})")
                time.sleep(1.0)

        self.cap = cap

        if not self.cap or not self.cap.isOpened():
            logger.warning(f"Kamera ({cam_id}) doğrudan açılamadı. Kullanılabilir kameralar taranıyor...")
            cams = CameraProcessor.scan_cameras()
            if cams and not getattr(self, '_fallback_attempted', False):
                self._fallback_attempted = True
                first_cam = cams[0]['id']
                logger.info(f"Takılı olan ilk kamera (ID: {first_cam}) otomatik seçilip başlatılıyor.")
                self.camera_id = first_cam
                return self._open_camera()
            logger.error(f"Kamera açılamadı: {cam_id}")
            return None
        self._fallback_attempted = False

        if isinstance(cam_id, str) and not str(cam_id).isdigit():
            try:
                import math
                fps_val = self.cap.get(cv2.CAP_PROP_FPS)
                if not fps_val or math.isnan(fps_val) or fps_val <= 0 or fps_val > 120:
                    fps_val = 25.0
                self.video_fps = fps_val
            except Exception:
                self.video_fps = 25.0
        else:
            self.video_fps = None

        try:
            width  = int(self.cfg.get('camera_width', 1280))
            height = int(self.cfg.get('camera_height', 720))
            fps_req = int(self.cfg.get('camera_fps', 30))

            if width > 0 and height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps_req > 0:
                self.cap.set(cv2.CAP_PROP_FPS, fps_req)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            logger.debug(f"Kamera prop ayarlama hatası (gözardı edildi): {e}")

        logger.info(f"Kamera açıldı: {cam_id} ({width}x{height} @ {fps_req} FPS MJPG)")
        return self.cap

    @staticmethod
    def get_camera_device_names() -> list:
        """Windows DirectShow veya Linux V4L2 üzerinden bağlı fiziki kamera cihaz isimlerini alır."""
        try:
            if sys.platform == 'win32':
                co_inited = False
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    co_inited = True
                except Exception:
                    try:
                        import comtypes
                        comtypes.CoInitialize()
                        co_inited = True
                    except Exception:
                        pass
                try:
                    from pygrabber.dshow_graph import FilterGraph
                    return [{'id': idx, 'name': dname, 'active': True} for idx, dname in enumerate(FilterGraph().get_input_devices())]
                finally:
                    if co_inited:
                        try:
                            import pythoncom
                            pythoncom.CoUninitialize()
                        except Exception:
                            try:
                                import comtypes
                                comtypes.CoUninitialize()
                            except Exception:
                                pass
            elif sys.platform.startswith('linux'):
                import glob
                devs = sorted(glob.glob('/dev/video*'))
                devices = []
                for d in devs:
                    idx_str = d.replace('/dev/video', '')
                    if idx_str.isdigit() and int(idx_str) % 2 == 0:
                        cam_id = int(idx_str)
                        devices.append({'id': cam_id, 'name': f"Kamera {cam_id} ({d})", 'active': True})
                return devices
        except Exception as e:
            logger.debug(f"Kamera cihaz isimleri alma hatası: {e}")
        return []

    @staticmethod
    def scan_cameras(max_index: int = 5) -> list:
        """Kullanılabilir fiziksel kameraları tarar ve cihaz detayları listesini döndürür."""
        devices = CameraProcessor.get_camera_device_names()
        if devices:
            return devices

        available = []
        import platform
        system = platform.system()
        for i in range(max_index):
            try:
                backend = cv2.CAP_DSHOW if system == 'Windows' else cv2.CAP_V4L2
                cap = cv2.VideoCapture(i, backend)
                if cap and cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append({'id': i, 'name': f"Kamera {i}", 'active': True})
                    cap.release()
            except Exception:
                pass

        return available

    def start_camera(self):
        if self.running or self.is_running:
            logger.info(f"Kamera {self.camera_id} başlatılırken mevcut aktif döngü sonlandırılıyor...")
            self.stop_camera()

        self.running = True
        self.is_running = True
        self._last_phone_seen_time = 0.0
        self._last_saved_durum = None
        self._last_saved_worker_id = None
        self._last_event_save_time = 0.0
        self._last_phone_boxes = []
        self._last_det_results = None
        self._cached_assigned_worker = None
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return True

    def stop_camera(self):
        self.running = False
        self.is_running = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        t_main = getattr(self, '_thread', None)
        t_ai   = getattr(self, '_ai_thread', None)
        self._thread = None
        self._ai_thread = None

        if t_main is not None and t_main.is_alive() and threading.current_thread() != t_main:
            try:
                t_main.join(timeout=0.8)
            except Exception:
                pass

        if t_ai is not None and t_ai.is_alive() and threading.current_thread() != t_ai:
            try:
                t_ai.join(timeout=0.8)
            except Exception:
                pass

        self._update_status({'running': False, 'durum': 'Kamera Kapalı'})



    def stop(self):
        self.stop_camera()

    def _ai_worker_loop(self):
        """Arka planda kamerayı asla yavaşlatmadan asenkron yapay zeka analizlerini yürütür."""
        logger.info("Arka plan Asenkron Yapay Zeka (AI Worker) thread'i başlatıldı.")
        ai_frame_count = 0
        kisi_takip = {}
        pixel_motion_thresh = float(self.cfg.get("pixel_motion_threshold", 6.0))
        hareket_esik_orani = float(self.cfg.get("hareket_esik_orani", 0.04))
        inaktif_limit = int(self.cfg.get("inaktif_kare_limiti", 450))
        last_save_time = time.time()



        while self.running:
            raw_frame = None
            with self._frame_lock:
                if hasattr(self, '_latest_raw_frame') and self._latest_raw_frame is not None:
                    raw_frame = self._latest_raw_frame.copy()

            if raw_frame is None:
                time.sleep(0.01)
                continue

            ai_frame_count += 1
            h, w = raw_frame.shape[:2]

            # 1. ROI Koordinatları Hesabı
            roi_sub = self.cfg.get('roi', {})
            x1_val = self.cfg.get('roi_x1', roi_sub.get('x1', roi_sub.get('x1_oran', 0.0)))
            y1_val = self.cfg.get('roi_y1', roi_sub.get('y1', roi_sub.get('y1_oran', 0.0)))
            x2_val = self.cfg.get('roi_x2', roi_sub.get('x2', roi_sub.get('x2_oran', 1.0)))
            y2_val = self.cfg.get('roi_y2', roi_sub.get('y2', roi_sub.get('y2_oran', 1.0)))

            try:
                x1_r, y1_r, x2_r, y2_r = float(x1_val), float(y1_val), float(x2_val), float(y2_val)
            except Exception:
                x1_r, y1_r, x2_r, y2_r = 0.0, 0.0, 1.0, 1.0

            if x1_r > 1.0: x1_r /= 100.0
            if y1_r > 1.0: y1_r /= 100.0
            if x2_r > 1.0: x2_r /= 100.0
            if y2_r > 1.0: y2_r /= 100.0

            roi_x1, roi_y1 = int(w * max(0.0, min(1.0, x1_r))), int(h * max(0.0, min(1.0, y1_r)))
            roi_x2, roi_y2 = int(w * max(0.0, min(1.0, x2_r))), int(h * max(0.0, min(1.0, y2_r)))

            # 2. Telefon Tespiti (YOLOv8 Det - Staggered Pipeline)
            phone_detected_raw = False
            phone_boxes_raw = []
            phone_conf_thresh = float(self.cfg.get('phone_conf', 0.25))

            det_imgsz = int(self.cfg.get('det_imgsz', 320))
            if self._det_model is not None and (ai_frame_count % 4 == 2 or not hasattr(self, '_last_det_results')):
                try:
                    self._last_det_results = self._det_model(raw_frame, conf=phone_conf_thresh, classes=[COCO_CELL_PHONE], imgsz=det_imgsz, verbose=False)
                except Exception as e:
                    logger.debug(f"AI Telefon tespit hatası: {e}")

            if hasattr(self, '_last_det_results') and self._last_det_results:
                for det_res in self._last_det_results:
                    if det_res.boxes is not None:
                        for box in det_res.boxes:
                            cls_id = int(box.cls[0])
                            conf_val = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
                            if cls_id == COCO_CELL_PHONE and conf_val >= phone_conf_thresh:
                                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                                bw = abs(bx2 - bx1)
                                bh = abs(by2 - by1)
                                area = bw * bh
                                if 15 <= bw <= 450 and 20 <= bh <= 550 and 300 <= area <= 150000:
                                    aspect_ratio = bh / float(bw) if bw > 0 else 0
                                    if 0.30 <= aspect_ratio <= 4.0:
                                        phone_boxes_raw.append((bx1, by1, bx2, by2))
                                        phone_detected_raw = True

            now_t = time.time()
            if phone_detected_raw:
                self._last_phone_seen_time = now_t

            is_phone_recent = (now_t - getattr(self, '_last_phone_seen_time', 0.0) < 1.5)
            phone_detected_in_roi = phone_detected_raw or is_phone_recent
            phone_boxes = phone_boxes_raw




            # 2.5 Kaynak Tespiti (YOLOv8 Det - Staggered Pipeline)
            welding_detected_raw = False
            welding_boxes_raw = []
            welding_conf_thresh = float(self.cfg.get('welding_conf', 0.30))

            weld_imgsz = int(self.cfg.get('welding_imgsz', 256))
            if self._welding_model is not None and (ai_frame_count % 4 == 3 or not hasattr(self, '_last_welding_results')):
                try:
                    self._last_welding_results = self._welding_model(raw_frame, conf=welding_conf_thresh, imgsz=weld_imgsz, verbose=False)
                except Exception as e:
                    logger.debug(f"AI Kaynak tespit hatası: {e}")

            if hasattr(self, '_last_welding_results') and self._last_welding_results:
                for w_res in self._last_welding_results:
                    if w_res.boxes is not None:
                        for box in w_res.boxes:
                            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0].cpu().numpy())
                            welding_boxes_raw.append((wx1, wy1, wx2, wy2))
                            welding_detected_raw = True

            welding_detected_in_roi = welding_detected_raw
            welding_boxes = welding_boxes_raw

            # 3. Poz & Hareket Tespiti (YOLOv8 Pose - Staggered Pipeline)
            aktif_kisi_var = False
            gorulen_kisi_id = set()
            kisi_sayisi_tespit = 0
            pose_labels = []

            roi_crop = raw_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_pixel_motion = 0.0
            if roi_crop.size > 0:
                gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.GaussianBlur(gray_roi, (15, 15), 0)
                if hasattr(self, '_prev_gray_roi') and self._prev_gray_roi is not None and self._prev_gray_roi.shape == gray_roi.shape:
                    diff = cv2.absdiff(gray_roi, self._prev_gray_roi)
                    roi_pixel_motion = float(np.mean(diff))
                self._prev_gray_roi = gray_roi

            if roi_pixel_motion >= pixel_motion_thresh:
                aktif_kisi_var = True

            if self._pose_model is not None and (ai_frame_count % 4 == 0 or not hasattr(self, '_last_pose_results')):
                try:
                    self._last_pose_results = self._pose_model(raw_frame, imgsz=256, verbose=False)
                except Exception as e:
                    logger.debug(f"AI Poz tespit hatası: {e}")

            skeletons = []
            person_track_list = []
            if hasattr(self, '_last_pose_results') and self._last_pose_results:
                for result in self._last_pose_results:
                    if result.keypoints is None: continue
                    kp_data = result.keypoints.data
                    if kp_data is None or len(kp_data) == 0: continue
                    kisi_sayisi_tespit = len(kp_data)

                    for kisi_idx in range(len(kp_data)):
                        kp = kp_data[kisi_idx].cpu().numpy()
                        visible_mask = kp[:, 2] > 0.3
                        visible_kps = kp[visible_mask, :2]
                        if len(visible_kps) < 3: continue
                        skeletons.append(kp)

                        gorulen_kisi_id.add(kisi_idx)
                        kisi_takip.setdefault(kisi_idx, {"prev": None, "inactive": 0})
                        durum_dict = kisi_takip[kisi_idx]

                        merkez = kp[0, :2] if kp[0, 2] > 0.3 else visible_kps.mean(axis=0)
                        kisi_in_roi = (roi_x1 <= merkez[0] <= roi_x2 and roi_y1 <= merkez[1] <= roi_y2)

                        top_y = max(25, int(visible_kps[:, 1].min()) - 15)
                        top_x = max(10, int(visible_kps[:, 0].mean()) - 60)

                        # Kişi Başı Özel Telefon Tespiti Eşleştirme (Konumsal/Mesafe Analizi)
                        px1 = visible_kps[:, 0].min() - 60
                        py1 = visible_kps[:, 1].min() - 60
                        px2 = visible_kps[:, 0].max() + 60
                        py2 = visible_kps[:, 1].max() + 80

                        kisi_has_phone = False
                        if phone_detected_in_roi and phone_boxes_raw:
                            for (pbx1, pby1, pbx2, pby2) in phone_boxes_raw:
                                phone_cx = (pbx1 + pbx2) / 2.0
                                phone_cy = (pby1 + pby2) / 2.0
                                if (px1 <= phone_cx <= px2 and py1 <= phone_cy <= py2) or \
                                   np.hypot(merkez[0] - phone_cx, merkez[1] - phone_cy) < 250:
                                    kisi_has_phone = True
                                    break


                        # Kişi Başı Özel Kaynak Tespiti Eşleştirme
                        kisi_is_welding = False
                        if welding_detected_in_roi and kisi_in_roi:
                            for (wx1, wy1, wx2, wy2) in welding_boxes_raw:
                                w_cx = (wx1 + wx2) / 2.0
                                w_cy = (wy1 + wy2) / 2.0
                                if (px1 <= w_cx <= px2 and py1 <= w_cy <= py2) or \
                                   np.hypot(merkez[0] - w_cx, merkez[1] - w_cy) < 250:
                                    kisi_is_welding = True
                                    break

                        if not kisi_in_roi or kisi_has_phone:
                            durum_dict["inactive"] = (inaktif_limit + 1) if kisi_has_phone else (durum_dict["inactive"] + 1)
                            durum_dict["prev"] = None
                            person_inact_sec = min(round(durum_dict["inactive"] / 30.0, 1), round(inaktif_limit / 30.0, 1))
                            person_track_list.append({
                                'idx': kisi_idx, 'kps': visible_kps, 'merkez': merkez,
                                'in_roi': kisi_in_roi, 'inactive_cnt': durum_dict["inactive"],
                                'inactive_sec': person_inact_sec, 'is_active': False,
                                'has_phone': kisi_has_phone, 'is_welding': False,
                                'top_x': top_x, 'top_y': top_y
                            })
                            continue

                        omuz_g = float(np.linalg.norm(kp[5, :2] - kp[6, :2])) if (kp[5, 2] > 0.3 and kp[6, 2] > 0.3) else 80.0
                        hareket_esik = max(omuz_g * hareket_esik_orani, 4.5)


                        is_person_moving = False
                        if durum_dict["prev"] is not None and durum_dict["prev"].shape == visible_kps.shape:
                            hareket = float(np.mean(np.linalg.norm(visible_kps - durum_dict["prev"], axis=1)))
                            if hareket > hareket_esik:
                                is_person_moving = True

                        if is_person_moving or kisi_is_welding:
                            durum_dict["inactive"] = 0
                            person_is_active = True
                            aktif_kisi_var = True
                        else:
                            durum_dict["inactive"] += 1
                            if durum_dict["inactive"] <= inaktif_limit:
                                person_is_active = True
                                aktif_kisi_var = True
                            else:
                                person_is_active = False

                        durum_dict["prev"] = visible_kps
                        person_inact_sec = min(round(durum_dict["inactive"] / 30.0, 1), round(inaktif_limit / 30.0, 1))
                        person_track_list.append({
                            'idx': kisi_idx, 'kps': visible_kps, 'merkez': merkez,
                            'in_roi': kisi_in_roi, 'inactive_cnt': durum_dict["inactive"],
                            'inactive_sec': person_inact_sec, 'is_active': person_is_active,
                            'has_phone': False, 'is_welding': kisi_is_welding,
                            'top_x': top_x, 'top_y': top_y
                        })

            # 4. İstasyon - Çalışan Otomatik Eşleştirmesi (Fotoğrafsız / Yüz Tanımasız)
            worker_id_detected, worker_name = self._get_station_worker()
            worker_confidence = 100.0
            face_boxes = []
            detected_workers = []

            for p in person_track_list:
                p['worker_id'] = worker_id_detected
                p['worker_name'] = worker_name
                detected_workers.append({
                    'id': worker_id_detected,
                    'name': worker_name,
                    'is_active': p['is_active'],
                    'has_phone': p.get('has_phone', False),
                    'is_welding': p.get('is_welding', False),
                    'inactive_sec': p['inactive_sec']
                })

            if not detected_workers:
                detected_workers = [{
                    'id': worker_id_detected,
                    'name': worker_name,
                    'is_active': aktif_kisi_var,
                    'has_phone': phone_detected_in_roi,
                    'is_welding': welding_detected_in_roi,
                    'inactive_sec': 0.0
                }]


            # Kişi Başı Özel Etiket ve Sayaç Hazırlığı
            limit_sec = round(inaktif_limit / 30.0, 1)
            for p in person_track_list:
                p_name = f"Çalışan-{p['idx'] + 1}"
                p_sec = p['inactive_sec']
                if not p['in_roi']:
                    lbl_txt = f"{p_name} [ROI DISI]"
                elif p.get('has_phone', False):
                    lbl_txt = f"{p_name} | TELEFON KULLANIMI!"
                elif p.get('is_welding', False):
                    lbl_txt = f"{p_name} | KAYNAK YAPIYOR!"
                elif p['is_active'] and p_sec == 0:
                    lbl_txt = f"{p_name} | AKTIF (0.0s)"
                else:
                    lbl_txt = f"{p_name} | SAYAC: {p_sec:.1f} / {limit_sec:.1f}s"

                pose_labels.append({
                    'text': lbl_txt,
                    'x': p['top_x'],
                    'y': p['top_y'],
                    'in_roi': p['in_roi'],
                    'is_active': (p['is_active'] and not p.get('has_phone', False)),
                    'has_phone': p.get('has_phone', False),
                    'is_welding': p.get('is_welding', False),
                    'inactive_sec': p_sec
                })

            # 5. Genel Durum Metni Belirleme
            any_phone = any(p.get('has_phone', False) for p in person_track_list) or phone_detected_in_roi
            any_welding = welding_detected_in_roi or any(p.get('is_welding', False) for p in person_track_list)

            now_t = time.time()
            if len(gorulen_kisi_id) > 0 or any_welding:
                self._last_worker_seen_time = now_t

            # 3 saniyelik tolerans/geçiş hafızası (Anlık poz kaçırmasında veya sabit kaynak duruşunda "İşçi Tespit Edilemedi" düşmesini engeller)
            is_recently_seen = (now_t - getattr(self, '_last_worker_seen_time', 0.0) < 3.0)

            herhangi_inaktif = any(not p.get('is_active', True) for p in person_track_list) if person_track_list else False
            herhangi_aktif = any(p.get('is_active', False) for p in person_track_list) if person_track_list else False

            if any_phone:
                genel_durum = DURUM_TELEFON
                genel_renk  = '#EF4444'
            elif any_welding:
                genel_durum = DURUM_KAYNAK
                genel_renk  = '#06B6D4'
            elif not (len(gorulen_kisi_id) > 0 or is_recently_seen):
                genel_durum = DURUM_TESPIT_YOK
                genel_renk  = '#888888'
            elif herhangi_inaktif:
                genel_durum = DURUM_INAKTIF
                genel_renk  = '#F59E0B'
            elif herhangi_aktif or is_recently_seen:
                genel_durum = DURUM_AKTIF
                genel_renk  = '#10B981'
            else:
                genel_durum = DURUM_TOLERANS
                genel_renk  = '#3B82F6'

            max_inact_cnt = max([d.get("inactive", 0) for d in kisi_takip.values()], default=0)
            inact_sec = min(round(max_inact_cnt / 30.0, 1), limit_sec)


            # Sonuçları Ana Kamera Thread'i ile Güvenli Paylaş
            with self._frame_lock:
                self._cached_ai_data = {
                    'phone_detected': phone_detected_in_roi,
                    'phone_detected_raw': phone_detected_raw,
                    'phone_boxes': phone_boxes,
                    'aktif_kisi_var': aktif_kisi_var,
                    'pose_labels': pose_labels,
                    'face_boxes': face_boxes,
                    'genel_durum': genel_durum,
                    'genel_renk': genel_renk,
                    'worker_name': worker_name,
                    'worker_id_detected': worker_id_detected,
                    'worker_confidence': worker_confidence,
                    'skeletons': skeletons,
                    'detected_workers': detected_workers,
                    'person_track_list': person_track_list,
                    'inactive_sec': inact_sec,
                    'inactive_limit_sec': limit_sec,
                    'kisi_cnt': max(len(gorulen_kisi_id), kisi_sayisi_tespit, len(detected_workers), 1 if worker_name else 0)
                }

            time.sleep(0.010)

    def run(self):
        self.running = True
        self.is_running = True
        self._update_status({'running': True})

        models_ok = self._load_models()
        cap = self._open_camera()

        if cap is None:
            self.running = False
            self.is_running = False
            self._update_status({'running': False, 'durum': 'Kamera Açılamadı'})
            return

        frame_count = 0
        fps = 0.0
        fps_timer = time.time()
        last_save_time = time.time()
        save_interval = int(self.cfg.get('save_interval', 5))

        self._cached_ai_data = {
            'phone_detected': False, 'phone_boxes': [], 'aktif_kisi_var': False,
            'pose_labels': [], 'face_boxes': [], 'genel_durum': DURUM_TESPIT_YOK,
            'genel_renk': '#888888', 'worker_name': '', 'worker_id_detected': None,
            'worker_confidence': 0.0, 'kisi_cnt': 0
        }

        # Arka Plan Asenkron AI Thread'ini Başlat
        self._ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self._ai_thread.start()

        logger.info("Ana Kamera Döngüsü (30-60 FPS Solid) başlatıldı.")

        while self.running:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(self.camera_id, str) and not str(self.camera_id).isdigit():
                    loop_video = self.cfg.get('loop_video', False)
                    if loop_video:
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        except Exception:
                            pass
                    else:
                        logger.info("Video dosyası sonuna ulaşıldı. Analiz tamamlandı.")
                        self.running = False
                        self.is_running = False
                        self._update_status({
                            'durum': 'Video Analizi Tamamlandı',
                            'running': False,
                            'video_completed': True
                        })
                        break

                if not ret or frame is None:
                    time.sleep(0.005)
                    continue


            # 0. Görüntü Ayarları
            if self.cfg.get('flip_h', False): frame = cv2.flip(frame, 1)
            if self.cfg.get('flip_v', False): frame = cv2.flip(frame, 0)
            brightness = int(self.cfg.get('brightness', 0))
            contrast   = float(self.cfg.get('contrast', 1.0))
            if brightness != 0 or contrast != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

            with self._frame_lock:
                self._latest_raw_frame = frame

            frame_count += 1
            h, w = frame.shape[:2]

            # Tam Donanım FPS Hesabı
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = now
                self._update_status({'fps': round(fps, 1)})

            annotated_frame = frame.copy()

            # AI Çıktılarını Güvenli Olarak Al ve Çiz
            with self._frame_lock:
                ai_data = dict(self._cached_ai_data)

            # 1. ROI Çizimi
            roi_sub = self.cfg.get('roi', {})
            x1_val = self.cfg.get('roi_x1', roi_sub.get('x1', roi_sub.get('x1_oran', 0.0)))
            y1_val = self.cfg.get('roi_y1', roi_sub.get('y1', roi_sub.get('y1_oran', 0.0)))
            x2_val = self.cfg.get('roi_x2', roi_sub.get('x2', roi_sub.get('x2_oran', 1.0)))
            y2_val = self.cfg.get('roi_y2', roi_sub.get('y2', roi_sub.get('y2_oran', 1.0)))

            try:
                x1_r, y1_r, x2_r, y2_r = float(x1_val), float(y1_val), float(x2_val), float(y2_val)
            except Exception:
                x1_r, y1_r, x2_r, y2_r = 0.0, 0.0, 1.0, 1.0

            if x1_r > 1.0: x1_r /= 100.0
            if y1_r > 1.0: y1_r /= 100.0
            if x2_r > 1.0: x2_r /= 100.0
            if y2_r > 1.0: y2_r /= 100.0

            roi_x1, roi_y1 = int(w * max(0.0, min(1.0, x1_r))), int(h * max(0.0, min(1.0, y1_r)))
            roi_x2, roi_y2 = int(w * max(0.0, min(1.0, x2_r))), int(h * max(0.0, min(1.0, y2_r)))

            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
            label_y = roi_y1 + 30 if roi_y1 + 30 < h else h - 10
            cv2.putText(annotated_frame, tr_to_ascii(f"CALISMA ALANI (ROI: Y1={int(y1_r*100)}% Y2={int(y2_r*100)}%)"),
                        (roi_x1 + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # 2. Telefon Kutuları Çizimi
            for (bx1, by1, bx2, by2) in ai_data.get('phone_boxes', []):
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(annotated_frame, "TELEFON!", (bx1, max(15, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # 3. İskelet Çizimi (Pose Keypoints & Kemik Çizgileri)
            p_boxes = ai_data.get('phone_boxes', [])
            for kp in ai_data.get('skeletons', []):
                for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
                    if pt1_idx < len(kp) and pt2_idx < len(kp):
                        if kp[pt1_idx, 2] > 0.48 and kp[pt2_idx, 2] > 0.48:
                            x1, y1 = int(kp[pt1_idx, 0]), int(kp[pt1_idx, 1])
                            x2, y2 = int(kp[pt2_idx, 0]), int(kp[pt2_idx, 1])
                            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                            # Telefon alanı içerisindeki veya çizgisindeki iskelet kemiklerini engelle
                            on_phone = False
                            for (pbx1, pby1, pbx2, pby2) in p_boxes:
                                margin = 30
                                if (pbx1 - margin <= x1 <= pbx2 + margin and pby1 - margin <= y1 <= pby2 + margin) or \
                                   (pbx1 - margin <= x2 <= pbx2 + margin and pby1 - margin <= y2 <= pby2 + margin) or \
                                   (pbx1 - margin <= mx <= pbx2 + margin and pby1 - margin <= my <= pby2 + margin):
                                    on_phone = True
                                    break
                            if not on_phone:
                                cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                for i in range(len(kp)):
                    if kp[i, 2] > 0.48:
                        kx, ky = int(kp[i, 0]), int(kp[i, 1])
                        on_phone = False
                        for (pbx1, pby1, pbx2, pby2) in p_boxes:
                            if pbx1 - 30 <= kx <= pbx2 + 30 and pby1 - 30 <= ky <= pby2 + 30:
                                on_phone = True
                                break
                        if not on_phone:
                            cv2.circle(annotated_frame, (kx, ky), 4, (0, 255, 0), -1)



            # 5. Genel Durum Rozeti Çizimi (Sol Üst)
            genel_durum = ai_data.get('genel_durum', DURUM_TESPIT_YOK)
            genel_renk  = ai_data.get('genel_renk', '#888888')
            overlay_bgr = self._hex_to_bgr(genel_renk)
            disp_status = tr_to_ascii(genel_durum)
            status_txt  = f"DURUM: {disp_status}"

            # Text boyutuna göre kutu genişliğini dinamik hesapla (Font ölçeği 0.65)
            font_scale = 0.65
            (dw, dh), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            durum_x1 = 15
            durum_y1 = 15
            durum_x2 = min(w - 15, durum_x1 + dw + 20)
            durum_y2 = 50

            cv2.rectangle(annotated_frame, (durum_x1, durum_y1), (durum_x2, durum_y2), (20, 20, 20), -1)
            cv2.rectangle(annotated_frame, (durum_x1, durum_y1), (durum_x2, durum_y2), overlay_bgr, 2)
            cv2.putText(annotated_frame, status_txt, (durum_x1 + 8, durum_y1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, overlay_bgr, 2, cv2.LINE_AA)

            # 6. Sağ Üst Kişi Başı Çoklu Sayaç Paneli (24/7 Canlı Saniye Sayacı)
            det_workers = ai_data.get('detected_workers', [])
            limit_sec = ai_data.get('inactive_limit_sec', 10.0)
            if det_workers:
                # Önce en geniş metni tespit et, çakışmayı kontrol et
                max_tw = 0
                worker_items = []
                for idx, w_info in enumerate(det_workers):
                    w_n = f"Çalışan-{idx+1}"
                    w_sec = w_info.get('inactive_sec', 0.0)
                    w_act = w_info.get('is_active', True)

                    if w_sec > 0:
                        t_color = (0, 0, 255) if not w_act else (0, 215, 255)
                        timer_str = tr_to_ascii(f"{w_n}: SAYAC {w_sec:.1f} / {limit_sec:.1f}s")
                    else:
                        t_color = (0, 255, 0)
                        timer_str = tr_to_ascii(f"{w_n}: AKTIF (0.0s)")

                    (tw, _), _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                    if tw > max_tw:
                        max_tw = tw
                    worker_items.append((timer_str, t_color, tw))

                tx1 = max(10, w - max_tw - 25)

                # Eğer sağ üstteki sayaç kutusu DURUM rozeti ile çakışıyorsa (dar ekran / dikey video), sayaç paneli DURUM rozetinin ALTINA kaydırılır
                if tx1 < durum_x2 + 15:
                    panel_y = durum_y2 + 10
                else:
                    panel_y = 15

                for timer_str, t_color, tw in worker_items:
                    curr_tx1 = max(10, w - tw - 25)
                    cv2.rectangle(annotated_frame, (curr_tx1 - 10, panel_y), (w - 15, panel_y + 35), (20, 20, 20), -1)
                    cv2.rectangle(annotated_frame, (curr_tx1 - 10, panel_y), (w - 15, panel_y + 35), t_color, 2)
                    cv2.putText(annotated_frame, timer_str, (curr_tx1, panel_y + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, t_color, 2, cv2.LINE_AA)
                    panel_y += 42

            with self._frame_lock:
                self._current_frame = annotated_frame

            status_payload = self._build_status(
                genel_durum, genel_renk, fps, ai_data.get('kisi_cnt', 0),
                ai_data.get('worker_name', ''), ai_data.get('worker_confidence', 0.0),
                ai_data.get('phone_detected', False)
            )
            self._update_status(status_payload)
            if self.socketio and frame_count % 5 == 0:
                try:
                    self.socketio.emit('status_update', status_payload)
                except Exception:
                    pass

            # ----------------------------------------------------------
            # 6. DB Kaydı & SocketIO Yayını (Sıfır Ham SQL - Tam ORM)
            # ----------------------------------------------------------
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                if last_save_time == 0 or (current_time - last_save_time > 15):
                    elapsed_seconds = save_interval
                else:
                    elapsed_seconds = min(10, max(1, int(round(current_time - last_save_time))))

                last_save_time = current_time
                now_dt = datetime.datetime.now()
                zaman_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                tarih_str = now_dt.strftime("%Y-%m-%d")



                phone_detected_in_roi = ai_data.get('phone_detected', False)
                phone_detected_raw = ai_data.get('phone_detected_raw', False)
                genel_durum = ai_data.get('genel_durum', DURUM_TESPIT_YOK)
                worker_id_detected = ai_data.get('worker_id_detected')
                worker_name = ai_data.get('worker_name', '')
                worker_confidence = ai_data.get('worker_confidence', 0.0)
                detected_workers = ai_data.get('detected_workers') or [{'id': worker_id_detected or 0, 'name': worker_name or 'Bilinmeyen Çalışan'}]

                # A) Her bir tespit edilen çalışan için Günlük Özet Tablosunu ORM ile Güncelle
                try:
                    with self.db_manager.get_session() as session:
                        for w_item in detected_workers:
                            w_id = w_item.get('id') or 0
                            w_name = w_item.get('name') or 'Bilinmeyen Çalışan'
                            w_active = w_item.get('is_active', True)

                            is_welding = w_item.get('is_welding', False) or (genel_durum == DURUM_KAYNAK)
                            is_phone = w_item.get('has_phone', False) or (phone_detected_raw and w_active)

                            if is_welding:
                                w_active = True

                            is_aktif = (w_active and not is_phone and not is_welding)
                            is_inaktif = (not w_active and not is_phone and not is_welding)

                            add_aktif = elapsed_seconds if is_aktif else 0
                            add_kaynak = elapsed_seconds if is_welding else 0
                            add_inaktif = elapsed_seconds if is_inaktif else 0
                            add_telefon = elapsed_seconds if is_phone else 0
                            add_ihlal = 1 if is_phone else 0

                            stmt = select(GunlukOzet).where(
                                GunlukOzet.tarih == tarih_str,
                                GunlukOzet.istasyon_adi == self._hostname,
                                GunlukOzet.worker_id == w_id
                            )
                            ozet = session.scalars(stmt).first()
                            if not ozet:
                                ozet = GunlukOzet(
                                    tarih=tarih_str,
                                    istasyon_adi=self._hostname,
                                    worker_id=w_id,
                                    worker_adi=w_name,
                                    toplam_aktif_sn=add_aktif,
                                    toplam_kaynak_sn=add_kaynak,
                                    toplam_inaktif_sn=add_inaktif,
                                    toplam_telefon_sn=add_telefon,
                                    telefon_ihlal_sayisi=add_ihlal,
                                    son_guncelleme=zaman_str
                                )
                                session.add(ozet)
                            else:
                                if w_name and w_name != 'Bilinmeyen Çalışan':
                                    ozet.worker_adi = w_name
                                ozet.toplam_aktif_sn += add_aktif
                                ozet.toplam_kaynak_sn += add_kaynak
                                ozet.toplam_inaktif_sn += add_inaktif
                                ozet.toplam_telefon_sn += add_telefon
                                ozet.telefon_ihlal_sayisi += add_ihlal
                                ozet.son_guncelleme = zaman_str
                except Exception as e:
                    logger.warning(f"Günlük özet ORM güncelleme hatası: {e}")


                # B) Sadece Durum veya Çalışan Değiştiğinde Event Kaydı At (ORM)
                is_video_mode = isinstance(self.camera_id, str) and not str(self.camera_id).isdigit()
                durum_degisti = (genel_durum != self._last_saved_durum)
                worker_degisti = (worker_id_detected != self._last_saved_worker_id)
                sure_doldu = (current_time - self._last_event_save_time >= (2 if is_video_mode else 10))

                if durum_degisti or worker_degisti or sure_doldu:
                    self._last_saved_durum = genel_durum
                    self._last_saved_worker_id = worker_id_detected
                    self._last_event_save_time = current_time

                    try:
                        with self.db_manager.get_session() as session:
                            for w_item in detected_workers:
                                w_id = w_item.get('id')
                                w_name = w_item.get('name') or ''
                                kayit = DurumKaydi(
                                    istasyon_adi=self._hostname,
                                    zaman=zaman_str,
                                    durum=genel_durum,
                                    worker_id=w_id,
                                    worker_adi=w_name,
                                    gonderildi=0
                                )
                                session.add(kayit)
                    except Exception as e:
                        logger.warning(f"Durum kaydı ORM hatası: {e}")

                if durum_degisti and (phone_detected_raw or genel_durum == DURUM_INAKTIF):
                    alarm_turu = 'TELEFON' if phone_detected_raw else 'HAREKETSİZLİK'
                    aciklama = f"{alarm_turu} tespit edildi ({self._hostname})"
                    try:
                        with self.db_manager.get_session() as session:
                            alarm_obj = Alarm(
                                istasyon_adi=self._hostname,
                                alarm_turu=alarm_turu,
                                aciklama=aciklama,
                                zaman=zaman_str,
                                okundu=0
                            )
                            session.add(alarm_obj)
                    except Exception as e:
                        logger.warning(f"Alarm kayıt ORM hatası: {e}")

                status_payload = self._build_status(
                    genel_durum, genel_renk, fps, ai_data.get('kisi_cnt', 0),
                    worker_name, worker_confidence, phone_detected_in_roi
                )
                self._update_status(status_payload)

                if self.socketio:
                    try:
                        self.socketio.emit('status_update', status_payload)
                    except Exception as e:
                        logger.debug(f"SocketIO emit hatası: {e}")

            # Video dosyası oynatılıyorsa gerçek zamanlı (1.0x FPS) kare pacing sağla
            if getattr(self, 'video_fps', None):
                target_delay = 1.0 / float(self.video_fps)
                proc_time = time.time() - loop_start
                sleep_time = target_delay - proc_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if cap:
            cap.release()
        self.running = False
        self.is_running = False
        self._update_status({'running': False})
        logger.info("Kamera döngüsü sonlandırıldı.")

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        hex_color = hex_color.lstrip('#')
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def _to_ascii(self, text: str) -> str:
        return tr_to_ascii(text)

    def _build_status(
        self, durum, renk, fps, kisi_sayisi,
        worker_name, worker_confidence, phone_detected
    ) -> dict:
        now_time = datetime.datetime.now().strftime('%H:%M:%S')
        actual_kisi_cnt = max(kisi_sayisi, 1 if worker_name else 0)

        conf_val = float(worker_confidence) if worker_confidence else 0.0
        if conf_val > 100:
            while conf_val > 100:
                conf_val /= 100.0
        elif 0 < conf_val <= 1.0:
            conf_val *= 100.0
        conf_fixed = round(conf_val, 1)

        return {
            'durum': durum,
            'status': durum,
            'renk': renk,
            'fps': round(fps, 1),
            'kisi_sayisi': actual_kisi_cnt,
            'person_count': actual_kisi_cnt,
            'istasyon': self._hostname,
            'station': self._hostname,
            'zaman': now_time,
            'last_update': now_time,
            'running': self.running,
            'camera_status': 'Kamera Çalışıyor' if self.running else 'Kamera Kapalı',
            'worker_name': worker_name,
            'worker_confidence': conf_fixed,
            'phone_detected': phone_detected,
            'camera_id': str(self.camera_id),
        }

    def _update_status(self, updates: dict):
        with self._status_lock:
            self._status.update(updates)
            self.current_status = self._status

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._current_frame is None:
                return None
            return self._current_frame.copy()

    def get_status(self) -> dict:
        with self._status_lock:
            return dict(self._status)

    def get_current_status(self) -> dict:
        return self.get_status()
