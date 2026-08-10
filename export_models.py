"""
YOLOv8 Modellerini ONNX Formatına Dönüştürme Betiği
Raspberry Pi 5 / ARM64 üzerinde 3x-4x daha yüksek FPS almak için çalıştırın.
Kullanım: python export_models.py
"""

import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

def export():
    pose_pt = BASE_DIR / 'yolov8n-pose.pt'
    det_pt  = BASE_DIR / 'yolov8n.pt'

    if pose_pt.exists():
        print(f"Poz modeli ONNX formatına aktarılıyor: {pose_pt}")
        model_pose = YOLO(str(pose_pt))
        model_pose.export(format='onnx', imgsz=320, simplify=True)
        print("yolov8n-pose.onnx başarıyla oluşturuldu.")

    if det_pt.exists():
        print(f"Tespit modeli ONNX formatına aktarılıyor: {det_pt}")
        model_det = YOLO(str(det_pt))
        model_det.export(format='onnx', imgsz=320, simplify=True)
        print("yolov8n.onnx başarıyla oluşturuldu.")

    weld_pt = BASE_DIR / 'welding_det.pt'
    if weld_pt.exists():
        print(f"Kaynak tespit modeli ONNX formatına aktarılıyor: {weld_pt}")
        model_weld = YOLO(str(weld_pt))
        model_weld.export(format='onnx', imgsz=320, simplify=True)
        print("welding_det.onnx başarıyla oluşturuldu.")


if __name__ == '__main__':
    export()
