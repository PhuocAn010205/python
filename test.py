import cv2
import imutils
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import csv
import os
import time
import pytesseract  # Ensure pytesseract is imported
import re  # Import regex for filtering alphanumeric characters

# Add Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tạo thư mục lưu hóa đơn nếu chưa có
if not os.path.exists("hoadon"):
    os.makedirs("hoadon")

# Load mô hình MobileNetV2 đã huấn luyện
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Danh sách nhãn xe
vehicle_labels = {
    'car': 'Xe con',
    'motorbike': 'Xe máy',
    'truck': 'Xe tải',
    'container': 'Container'
}

# Mức phí cho từng loại xe
VEHICLE_FEES = {
    'car': 30000,
    'motorbike': 0,
    'truck': 50000,
    'container': 70000
}

def classify_vehicle(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]

    if 'truck' in decoded:
        return 'truck'
    elif 'container' in decoded or 'trailer' in decoded:
        return 'container'
    elif 'motorbike' in decoded or 'moped' in decoded:
        return 'motorbike'
    elif 'car' in decoded or 'sedan' in decoded:
        return 'car'
    else:
        return 'unknown'

def infer_vehicle_from_plate_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    yellow_ratio = cv2.countNonZero(mask_yellow) / (roi.shape[0] * roi.shape[1])

    if yellow_ratio > 0.3:
        return "truck"  # biển vàng → xe tải, dịch vụ
    else:
        return "car"    # biển trắng → xe con

def extract_license_plate_text(roi):
    """Extract alphanumeric text from the license plate region using OCR."""
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_roi, config='--psm 7')  # PSM 7 for single text line
    print("Raw OCR Output:", text)  # Debugging the OCR output
    filtered_text = re.sub(r'[^A-Za-z0-9]', '', text)  # Keep only alphanumeric characters
    return filtered_text.strip()

def detect_plate_and_classify(path):
    print("Đang xử lý ảnh:", path)  # Debugging
    image = cv2.imread(path)
    if image is None:
        return "Lỗi: Không đọc được ảnh. Vui lòng kiểm tra đường dẫn hoặc định dạng ảnh."
    image_resized = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 50, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

    plate_found = False
    roi = image_resized
    plate_text = "Không xác định"
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:  # Tìm vùng có 4 cạnh
            x, y, w, h = cv2.boundingRect(approx)
            roi = image_resized[y:y + h, x:x + w]
            # cv2.imwrite("debug_plate.jpg", roi)  # Lưu vùng biển số để kiểm tra
            plate_text = extract_license_plate_text(roi)
            print("Biển số phát hiện:", plate_text)  # Debugging
            plate_found = True
            break  # Dừng vòng lặp ngay khi tìm thấy biển số

    if plate_found:
        vehicle_type = infer_vehicle_from_plate_color(roi)
    else:
        vehicle_type = classify_vehicle(roi)

    label = vehicle_labels.get(vehicle_type, "Không rõ")
    fee = VEHICLE_FEES.get(vehicle_type, 0)

    try:
        with open("hoadon/hoadon.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([label, fee, plate_text])
    except Exception as e:
        print(f"Lỗi khi ghi file CSV: {e}")

    return f"✅ Phát hiện {label}. Phí thu: {fee:,} VNĐ. Biển số: {plate_text}"

def process_folder_images(folder_path):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_ext):
            image_path = os.path.join(folder_path, filename)
            result = detect_plate_and_classify(image_path)
            print(f"{filename}: {result}")
            time.sleep(0.5)
            os.remove(image_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Hệ thống thu phí tự động thông minh")
        self.image_panel = None

        self.label = Label(root, text="Chọn ảnh hoặc thư mục để quét", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_upload = Button(root, text="📁 Chọn ảnh đơn", command=self.upload_image)
        self.btn_upload.pack(pady=5)

        self.btn_folder = Button(root, text="🗂 Quét toàn bộ ảnh trong thư mục", command=self.process_folder)
        self.btn_folder.pack(pady=5)

        # Set the result label text color to black
        self.result_label = Label(root, text="", font=("Arial", 12), fg="black")
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)

        if self.image_panel is None:
            self.image_panel = Label(image=img)
            self.image_panel.image = img
            self.image_panel.pack()
        else:
            self.image_panel.configure(image=img)
            self.image_panel.image = img

        result = detect_plate_and_classify(file_path)
        self.result_label.config(text=result, fg="black")  # Ensure text color is black

    def process_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        process_folder_images(folder_path)
        self.result_label.config(text="✅ Đã xử lý toàn bộ ảnh trong thư mục!", fg="black")  # Ensure text color is black

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    print("✅ Hoàn tất xử lý ảnh!")
    print("📁 Hóa đơn đã lưu trong thư mục 'hoadon/'")
