import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO  # Make sure you installed ultralytics: pip install ultralytics

# -----------------------------
# Load your trained YOLO model
# -----------------------------
model = YOLO("best.pt")   # replace "best.pt" with the path to your trained YOLO weights

# -----------------------------
# Configure Intel RealSense
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB + Depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # -----------------------------
        # Run YOLO inference on RGB
        # -----------------------------
        results = model(color_image, verbose=False)

        # Get detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Class + confidence
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                # Center point for depth lookup
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Get depth (in meters) at center
                depth = depth_frame.get_distance(cx, cy)

                # Draw bounding box + label
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"{label} {conf:.2f} {depth:.2f}m",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        # Show the result
        cv2.imshow("YOLO + RealSense", color_image)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
