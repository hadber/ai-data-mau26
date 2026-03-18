from ultralytics import YOLO

def train():
	# Load a pretrained YOLO26 model
	model = YOLO("yolo26n.pt")
	# Train on coco dataset
	results = model.train(data="custom.yaml", epochs=25, imgsz=640, save=True)

if __name__ == "__main__":
	train()
