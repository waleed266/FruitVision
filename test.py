import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# -----------------------------
# Load YOLO model (segmentation enabled)
# -----------------------------
model = YOLO("best.pt")  # replace with your trained weights

# -----------------------------
# Configure Intel RealSense
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)

# -----------------------------
# Draw professional box function
# -----------------------------
def draw_box(img, x1, y1, x2, y2, label, conf, depth, color=(0, 255, 0)):
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {conf:.2f} {depth:.2f}m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), font_thickness)

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Brighten RGB camera view
        color_image = cv2.convertScaleAbs(color_image, alpha=1.25, beta=20)

        # -----------------------------
        # Run YOLO segmentation
        # -----------------------------
        results = model(color_image, stream=True)  # stream=True enables segmentation overlays
        for r in results:
            # r.masks contains segmentation masks
            # r.plot() draws both masks and boxes
            annotated_frame = r.plot()

            # Draw professional bounding boxes with depth
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    # Get depth at center
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    depth = depth_frame.get_distance(cx, cy)

                    draw_box(annotated_frame, x1, y1, x2, y2, label, conf, depth)

        # -----------------------------
        # Split screen: RGB with segmentation | Depth
        # -----------------------------
        combined = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow("YOLOv8 Segmentation + RealSense Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
