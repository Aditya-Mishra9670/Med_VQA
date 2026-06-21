import pandas as pd
from pathlib import Path

df = pd.read_csv("data/VQA-Med-2019-master/valdf.csv")

img_dir = Path("data/VQA-Med-2019-master/val_images")

missing = []

for img_id in df["img_id"]:
    if not (img_dir / f"{img_id}.jpg").exists():
        missing.append(img_id)

print("Missing images:", len(missing))

for x in missing[:50]:
    print(x)