Sure — here’s a simpler, plain-text version of your **README**:

---

# Fruit Segmentation Project (YOLOv8)

This project is about detecting and segmenting fruits using the YOLOv8 model.
The model can recognize and segment five fruit types: **Apple, Banana, Orange, Avocado, and Strawberry**.

---

## Dataset

The dataset was created by combining COCO fruit images (Apple, Banana, Orange) with two custom annotated datasets (Avocado and Strawberry).
All images were resized to **640x640** and annotated in **YOLO format**.
Augmentation such as rotation, brightness, exposure, and saturation changes were applied to improve the model’s performance.

Dataset link:
[Google Drive Dataset](https://drive.google.com/drive/folders/1Nj2GMEmXtMuxexqPpWK7nSsu90pCK_sl?usp=drive_link)

---

## Model Training

The model was trained on **Google Colab** using YOLOv8.
Main training details:

* Image size: 640
* Epochs: 100
* Optimizer: Adam
* Validation split: 10%
* Dataset size: around 5000+ images

---

## Evaluation

The model was evaluated using **mAP**, **precision**, and **recall**.
Results showed good detection and segmentation accuracy for all five fruit classes.

---

## Usage

1. Install requirements:

   ```
   pip install ultralytics opencv-python numpy matplotlib
   ```
2. Train the model:

   ```
   yolo task=segment mode=train data=data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
   ```
3. Run inference:

   ```
   yolo task=segment mode=predict model=best.pt source=your_test_images/
   ```

---

## Credits

Developed by **Ahmed [Your Last Name]**
Supervised by **Dr. Xing Wang**

Special thanks to **Roboflow** for data annotation tools and **Ultralytics YOLO** for the training framework.

---

Would you like me to make this one even shorter (like a few paragraphs only) for quick README display on GitHub, or keep this version with basic sections?
