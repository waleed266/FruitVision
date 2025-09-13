import sys
import cv2
from ultralytics import YOLO

YOLO_MODEL_PATH = "best.pt"  # Your trained model

def image_detection(image_path):
    model = YOLO(YOLO_MODEL_PATH)  # Load YOLO model
    
    # Step 1: Read the image
    img = cv2.imread(image_path)
    
    # Step 2: Preprocess the image (Optional)
    # Enhance contrast and sharpness
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)  # Adjust contrast and brightness
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Smooth out the image (optional)

    # Resize the image (optional, try different sizes)
    img_resized = cv2.resize(img, (640, 640))  # Change to the size expected by YOLO model
    
    # Step 3: Run YOLO detection with adjusted settings
    results = model(img_resized, conf=0.4, iou=0.5)  # Adjust confidence (conf) and IoU thresholds
    
    # Step 4: Annotate the image with detection results
    annotated_frame = results[0].plot()  # Annotate image with detection boxes
    
    # Step 5: Display the annotated image
    cv2.imshow("Image Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get image file path passed from the GUI
    image_detection(image_path)
