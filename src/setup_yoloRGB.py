import os
import random
import shutil
import yaml
from pathlib import Path


def setup_yoloRGB(
    patches_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    classes=None
):
    """
    Suddivide le immagini e le etichette YOLO in train/val/test e crea un file data.yaml.
    
    :param patches_dir: cartella che contiene le sottocartelle 'png' e 'labels'
    :param output_dir: cartella in cui creare le sottocartelle train/val/test e il file data.yaml
    :param train_ratio: proporzione immagini train (default: 0.7)
    :param val_ratio: proporzione immagini val (default: 0.2)
    :param test_ratio: proporzione immagini test (default: 0.1)
    :param classes: lista dei nomi delle classi
    """

    png_dir = os.path.join(patches_dir, "png")
    label_dir = os.path.join(patches_dir, "labels")
    assert os.path.isdir(png_dir) and os.path.isdir(label_dir), "Cartelle png/ e labels/ non trovate."

    # === Crea output dir ===
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # === Filtra immagini con etichette ===
    images = [f for f in os.listdir(png_dir) if f.endswith(".png")]
    valid_images = [f for f in images if os.path.exists(os.path.join(label_dir, f.replace('.png', '.txt')))]
    print(f"✔️ Trovate {len(valid_images)} immagini con etichette.")

    # === Mescola e divide ===
    random.shuffle(valid_images)
    n = len(valid_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': valid_images[:n_train],
        'val': valid_images[n_train:n_train + n_val],
        'test': valid_images[n_train + n_val:]
    }

    for split, files in splits.items():
        for f in files:
            src_img = os.path.join(png_dir, f)
            src_lbl = os.path.join(label_dir, f.replace(".png", ".txt"))

            dst_img = os.path.join(output_dir, split, 'images', f)
            dst_lbl = os.path.join(output_dir, split, 'labels', f.replace(".png", ".txt"))

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

        print(f"📂 {split}: {len(files)} immagini copiate")

    # === Scrivi data.yaml ===
    yaml_dict = {
        'train': os.path.abspath(os.path.join(output_dir, 'train', 'images')),
        'val': os.path.abspath(os.path.join(output_dir, 'val', 'images')),
        'test': os.path.abspath(os.path.join(output_dir, 'test', 'images')),
        'nc': len(classes) if classes else 1,
        'names': classes if classes else ["class0"]
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f)

    print(f"✅ File data.yaml creato in: {yaml_path}")