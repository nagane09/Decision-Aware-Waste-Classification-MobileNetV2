import os
import shutil
import random


raw_dir = "dataset/raw"
dataset_dir = "dataset/dataset"

classes = ["cardboard", "glass", "metal", "plastic"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for split in ["train", "validation", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(raw_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(dataset_dir, "train", cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(dataset_dir, "validation", cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(dataset_dir, "test", cls, img))

print("Dataset split completed!")
