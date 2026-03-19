from os import listdir
from os.path import isfile, join
from sys import argv

from ultralytics import YOLO


def benchmark(model_path, images_path, repetitions):
	model = YOLO(model_path)
	inference_time = []
	
	for i in range(repetitions):
		for img in listdir(images_path):
			result = model(f"{images_path}/{img}")
			inference_time.append(result[0].speed['inference'])

	return inference_time

if __name__ == "__main__":

	if len(argv) < 1:
		print('you need to provide the output filename')
		exit()

	hardware = argv[1]
	
	repetitions = 5
	
	with open(f"benchmark_{hardware}_full", 'w') as f:
		f.write(','.join([str(x) for x in benchmark("best.onnx", "img", 5)]))
	
	with open(f"benchmark_{hardware}_half", 'w') as f:
		f.write(','.join([str(x) for x in benchmark("best_half.onnx", "img", 5)]))

#benchmark("best.onnx", "img", 10)

