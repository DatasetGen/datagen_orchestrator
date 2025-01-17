from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
train_results = model.train(data="./yolo_dataset/qaisc.yaml", epochs=1000, imgsz=640)
validation_results = model.val(data="./yolo_dataset/qaisc.yaml", imgsz=640)
