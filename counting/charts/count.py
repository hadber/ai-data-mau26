from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import csv

data_path = "datasets/kitti/labels/train/"

people_dict = {}
for f in listdir(data_path):
	file_path = join(data_path, f)
	if not isfile(file_path):
		continue

	people_count = 0
	# our people labels are:
	# 3: pedestrian
	# 4: Person_sitting
	with open(file_path) as _file:
		for line in _file.readlines():
			label = line.split()[0]
			people_count += 1 if label in ['3', '4'] else 0

	if people_count in people_dict:
		people_dict[people_count] += 1
	else:
		people_dict[people_count] = 1

x, y = list(people_dict.keys()), list(people_dict.values())

plt.figure(figsize=(8,5))
plt.bar(x, y)
plt.xticks(range(len(x)))
plt.xlabel("Number of persons in image")
plt.ylabel("Number of images")
plt.title(f"KITTI Dataset - Person Distribution ({len(listdir(data_path))} samples)")
plt.show()

with open("results.csv", "w", newline="") as f:
	w = csv.DictWriter(f, people_dict.keys())
	w.writeheader()
	w.writerow(people_dict)
