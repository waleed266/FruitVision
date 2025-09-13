import sys
import cv2
from ultralytics import YOLO

YOLO_MODEL_PATH = "best.pt"  # Your trained model

def image_detection(image_path):
    model = YOLO(YOLO_MODEL_PATH)  # Load YOLO model
    
    # Step 1: Read the image
    img = cv2.imread(image_path)
    
    # Step 2: Run YOLO detection (no extra preprocessing or adjustments)
    results = model(img)
    
    # Step 3: Annotate the image with detection results
    annotated_frame = results[0].plot()  # Annotate image with detection boxes
    
    # Step 4: Display the annotated image
    cv2.imshow("Image Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get image file path passed from the GUI
    image_detection(image_path)
