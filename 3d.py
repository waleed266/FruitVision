import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import open3d as o3d

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("best.pt")  # Replace with your YOLO weights

# -----------------------------
# Configure RealSense
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

colorizer = rs.colorizer()
depth_intrin = None

# -----------------------------
# Open3D visualization
# -----------------------------
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window("3D Point Cloud", width=640, height=480)
vis.add_geometry(pcd)

pc = rs.pointcloud()

# -----------------------------
# Smooth coordinates storage
# -----------------------------
prev_coords = {}

def smooth_coords(label, coords, alpha=0.5):
    if label in prev_coords:
        new_coords = alpha * np.array(coords) + (1 - alpha) * np.array(prev_coords[label])
        prev_coords[label] = new_coords
        return new_coords
    else:
        prev_coords[label] = coords
        return coords

# -----------------------------
# Average depth over bounding box
# -----------------------------
def get_avg_depth(depth_frame, cx, cy, size=5):
    x_min = max(cx - size//2, 0)
    x_max = min(cx + size//2, depth_frame.get_width() - 1)
    y_min = max(cy - size//2, 0)
    y_max = min(cy + size//2, depth_frame.get_height() - 1)
    
    depth_values = []
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            d = depth_frame.get_distance(x, y)
            if d > 0:  # valid depth
                depth_values.append(d)
    if len(depth_values) == 0:
        return 0
    return float(np.mean(depth_values))

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        if depth_intrin is None:
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # -----------------------------
        # YOLO detection
        # -----------------------------
        results = model(color_image, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Fine-tuned depth
                avg_depth = get_avg_depth(depth_frame, cx, cy, size=7)
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], avg_depth)
                X, Y, Z = smooth_coords(label, [X, Y, Z])

                # Draw box and 3D coordinates
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image,
                            f"{label} {conf:.2f} {avg_depth:.2f}m ({X:.2f},{Y:.2f},{Z:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        # -----------------------------
        # Update 3D point cloud
        # -----------------------------
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)

        vertices = np.asanyarray(points.get_vertices())
        if len(vertices) > 0:
            vtx = np.stack([vertices['f0'], vertices['f1'], vertices['f2']], axis=-1).astype(np.float32)
            vtx = vtx[::5]  # optional downsample

            color_image_flat = np.asanyarray(color_frame.get_data())
            colors = (color_image_flat.reshape(-1,3)/255.0).astype(np.float32)

            pcd.points = o3d.utility.Vector3dVector(vtx)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        # -----------------------------
        # Split-screen RGB + Depth
        # -----------------------------
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("YOLO + RealSense 3D (RGB | Depth)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
