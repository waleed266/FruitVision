import sys
import cv2
from ultralytics import YOLO

YOLO_MODEL_PATH = "best.pt"  # your trained model

def image_detection(image_path):
    model = YOLO(YOLO_MODEL_PATH)  # Load YOLO model
    img = cv2.imread(image_path)
    results = model(img)
    annotated_frame = results[0].plot()  # Annotate image with detection
    cv2.imshow("Image Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get image file path passed from the GUI
    image_detection(image_path)
