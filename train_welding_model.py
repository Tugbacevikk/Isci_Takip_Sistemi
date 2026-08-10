import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

def train_welding():
    base_dir = Path(__file__).parent.resolve()
    dataset_dir = base_dir / "fire- welding detection.v1i.yolov8" / "fire- welding detection.v1i.yolov8"
    
    if not dataset_dir.exists():
        print(f"Hata: Veri seti dizini bulunamadı: {dataset_dir}")
        sys.exit(1)
        
    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        print(f"Hata: data.yaml bulunamadı: {data_yaml_path}")
        sys.exit(1)
        
    print(f"Veri seti dizini: {dataset_dir}")
    
    # Update data.yaml with absolute paths for robustness
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        
    data_cfg['train'] = str((dataset_dir / "train" / "images").resolve())
    data_cfg['val']   = str((dataset_dir / "valid" / "images").resolve())
    data_cfg['test']  = str((dataset_dir / "test" / "images").resolve())
    
    temp_yaml_path = dataset_dir / "data_abs.yaml"
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_cfg, f)
        
    print("YOLOv8 Kaynak Tespiti modeli eğitimi başlatılıyor (15 epoch)...")
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(temp_yaml_path),
        epochs=15,
        imgsz=416,
        batch=16,
        project=str(base_dir / "scratch" / "welding_runs"),
        name="welding_model",
        exist_ok=True,
        verbose=True
    )
    
    best_weights = base_dir / "scratch" / "welding_runs" / "welding_model" / "weights" / "best.pt"
    target_weights = base_dir / "welding_det.pt"
    
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"\n[BAŞARILI] Kaynak tespit modeli kaydedildi: {target_weights}")
    else:
        print("\n[UYARI] best.pt bulunamadı, eğitim çıktısı kontrol edilmeli.")

if __name__ == "__main__":
    train_welding()
