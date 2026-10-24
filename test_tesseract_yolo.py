import cv2
import pytesseract
from ultralytics import YOLO

print("[INFO] YOLOv8 Model load ho raha hai...")
yolo_model = YOLO('yolov8n.pt')

print("[INFO] Webcam open ho raha hai...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Webcam open nahi ho saka.")
    exit()

print("\n=== CONTROLS ===")
print("Press 't' -> Fast Text Read (Tesseract OCR)")
print("Press 'q' -> Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Real-time YOLO Object Detection
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Blind Assistance - Tesseract Mode", annotated_frame)
    key = cv2.waitKey(1) & 0xFF

    # Fast OCR Trigger (press 't')
    if key == ord('t'):
        print("\n[OCR] Processing image with Tesseract...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        
        cleaned_text = text.strip()
        if cleaned_text:
            print("=" * 40)
            print(f"[OCR RESULT]:\n{cleaned_text}")
            print("=" * 40)
        else:
            print("[OCR RESULT]: Koi clear text nahi mila. Camera ke aage text sahi se rakhein.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
