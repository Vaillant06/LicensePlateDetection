import cv2
from ultralytics import YOLO

model = YOLO("lp_detect/exp1/weights/best.pt")

url = "http://192.168.29.38:8080/stream.mjpeg"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Unable to open MJPEG stream")
    exit()

print("🎥 Streaming started... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("License Plate Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
