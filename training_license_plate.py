from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    imgsz=512,
    epochs=50,
    batch=8,
    device=0,
    workers=4,
    verbose=True,
    project="lp_detect",
    name="exp1",
    patience=15,
    cos_lr=True,
)
