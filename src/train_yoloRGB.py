import os
import yaml
from ultralytics import YOLO

def train_yoloRGB(
    train_images_dir,
    val_images_dir,
    classes,
    output_dir,
    pretrained_model_path=None,
    which_model='yolov11m.pt',
    model_name='yolo_model',
    epochs=100,
    imgsz=512,
    batch=64,
    learning_rate=1e-4,
    patience=20,
    copy_paste=0.0
):
    """
    Addestra un modello YOLOv8 con Ultralytics.

    Parametri obbligatori:
    - train_images_dir: path alle immagini di training
    - val_images_dir: path alle immagini di validazione
    - classes: lista di nomi delle classi
    - output_dir: directory dove salvare i risultati

    Parametri opzionali:
    - pretrained_model_path: path a modello .pt già addestrato
    - which_model: modello di base (es. 'yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt') se non si usa pretrained
    - model_name: nome della run
    - altri iperparametri
    """

    # === Scegli il modello da usare ===
    if pretrained_model_path:
        model = YOLO(pretrained_model_path)
        print(f"ℹ️ Caricato modello preaddestrato: {pretrained_model_path}")
    else:
        model = YOLO(which_model)
        print(f"⚠️ Nessun modello preaddestrato fornito. Caricato modello base: {which_model}")

    # === Crea file data.yaml dinamico ===
    os.makedirs(output_dir, exist_ok=True)
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    data_config = {
        'train': train_images_dir,
        'val': val_images_dir,
        'names': classes,
        'nc': len(classes)
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    # === Addestramento ===
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=learning_rate,
        name=model_name,
        patience=patience,
        copy_paste=copy_paste,
        project=output_dir
    )

    # === Percorso modello finale ===
    best_model_path = os.path.join(output_dir, model_name, 'weights', 'best.pt')
    print(f"✅ Addestramento completato. Modello salvato in: {best_model_path}")

    return best_model_path