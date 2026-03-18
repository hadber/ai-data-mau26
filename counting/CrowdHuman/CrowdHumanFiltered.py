
!pip install ultralytics

from ultralytics import YOLO
import os
import shutil
from tqdm import tqdm


from google.colab import drive
drive.mount('/content/drive')


INPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/AI&DataProject/Images"


OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/CrowdHuman_filtered"


IMG_OUT = os.path.join(OUTPUT_DIR, "images")
LBL_OUT = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

model = YOLO("yolo26n.pt")  


MIN_PEOPLE = 5
MAX_PEOPLE = 15


image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Total images found: {len(image_files)}")

for img_name in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, img_name)

    try:
        results = model(img_path, verbose=False)[0]
    except:
        continue

    boxes = results.boxes
    if boxes is None:
        continue

    
    person_boxes = [b for b in boxes if int(b.cls) == 0]
    num_people = len(person_boxes)

    if MIN_PEOPLE <= num_people <= MAX_PEOPLE:

        
        shutil.copy(img_path, os.path.join(IMG_OUT, img_name))

       
        h, w = results.orig_shape
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(LBL_OUT, label_name)

        with open(label_path, 'w') as f:
            for box in person_boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"0 {xc} {yc} {bw} {bh}\n")


FINAL_DIR = os.path.join(OUTPUT_DIR, "dataset")

TRAIN_IMG = os.path.join(FINAL_DIR, "train/images")
TRAIN_LBL = os.path.join(FINAL_DIR, "train/labels")

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)


for file in os.listdir(IMG_OUT):
    shutil.move(os.path.join(IMG_OUT, file), os.path.join(TRAIN_IMG, file))


for file in os.listdir(LBL_OUT):
    shutil.move(os.path.join(LBL_OUT, file), os.path.join(TRAIN_LBL, file))


yaml_content = f"""
path: {FINAL_DIR}
train: train/images
val: train/images

names:
  0: person
"""

with open(os.path.join(FINAL_DIR, "dataset.yaml"), "w") as f:
    f.write(yaml_content)


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("person_distribution_filtered.csv")


plt.figure(figsize=(10, 6))
plt.bar(df["people_in_image"], df["number_of_images"], color='orange')


plt.xlabel("Number of People in Image")
plt.ylabel("Number of Images")
plt.title("Distribution of People per Image")
plt.xticks(df["people_in_image"])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("person_distribution_filtered_plot.png")
print("Plot saved as person_distribution_filtered_plot.png")
print("DONE!")
print(f"Dataset saved at: {FINAL_DIR}")
