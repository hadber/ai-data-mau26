
import os
import pandas as pd
from ultralytics import YOLO

from google.colab import drive
drive.mount('/content/drive')


image_folder = "/content/drive/MyDrive/Colab Notebooks/AI&DataProject/Images"   # change this to your folder
model = YOLO("yolo26n.pt")


people_distribution = {}


for filename in os.listdir(image_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(image_folder, filename)

        results = model(path)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if r.names[cls] == "person":
                    person_count += 1

        
        if person_count not in people_distribution:
            people_distribution[person_count] = 0

        people_distribution[person_count] += 1

distribution_df = pd.DataFrame(
    sorted(people_distribution.items()),
    columns=["people_in_image", "number_of_images"]
)

distribution_df.to_csv("person_distribution.csv", index=False)

print("CSV saved as person_distribution.csv")
print(distribution_df)
