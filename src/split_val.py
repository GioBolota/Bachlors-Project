"""
to split 10 percent of training data into a val/ .
only need to run once: python src/split_val.py
"""
import os, shutil, random

random.seed(42)

for cls in ["real", "fake"]:
    src_dir = os.path.join("data", "train", cls)
    val_dir = os.path.join("data", "val", cls)
    os.makedirs(val_dir, exist_ok=True)

    images = os.listdir(src_dir)
    random.shuffle(images)
    val_images = images[:int(len(images) * 0.1)]

    for fname in val_images:
        shutil.move(
            os.path.join(src_dir, fname),
            os.path.join(val_dir, fname)
        )
    print(f"{cls}: moved {len(val_images)} images to val/")

print("Done. data/ now has train/, val/, and test/.")