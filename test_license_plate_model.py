from ultralytics import YOLO
import cv2
import os

model = YOLO("lp_detect/exp1/weights/best.pt") 

test_path = "test_images" 

os.makedirs("results", exist_ok=True)

def run_inference(path):
    project = "results"
    name = "preds"

    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for file in files:
            img_path = os.path.join(path, file)
            results = model(img_path, save=True, project=project, name=name, exist_ok=True)
            print(f"Processed: {img_path}")
    else:
        results = model(path, save=True, project=project, name=name, exist_ok=True)
        print(f"Processed: {path}")


if __name__ == "__main__":
    run_inference(test_path)
    print("Done! Results saved in: results/preds")
