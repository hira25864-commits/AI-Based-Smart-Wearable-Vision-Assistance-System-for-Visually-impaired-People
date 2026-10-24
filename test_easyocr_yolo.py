import cv2
import easyocr
import pyttsx3
import threading
from ultralytics import YOLO

# Audio Engine Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

print("[INFO] EasyOCR reader load ho raha hai (is me thoda time lagega)...")
reader = easyocr.Reader(['en'], gpu=False)

print("[INFO] YOLOv8 Model load ho raha hai...")
yolo_model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam open nahi ho saka.")
    exit()

speak("Easy OCR System ready. Press E for Easy OCR, O for objects.")

print("\n=== CONTROLS ===")
print("Press 'e' -> Read Text with EasyOCR (+ Voice)")
print("Press 'o' -> Announce Detected Objects (+ Voice)")
print("Press 'q' -> Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("EasyOCR Assistance System", annotated_frame)
    key = cv2.waitKey(1) & 0xFF

    # EasyOCR Trigger (press 'e')
    if key == ord('e'):
        print("\n[EasyOCR] Processing image... (is me 5-10 sec lag sakte hain)")
        speak("Reading text, please wait.")
        
        # EasyOCR text extraction
        ocr_results = reader.readtext(frame)
        extracted_text = " ".join([text[1] for text in ocr_results]).strip()
        
        if extracted_text:
            print("=" * 40)
            print(f"[EasyOCR RESULT]:\n{extracted_text}")
            print("=" * 40)
            speak(f"Text detected: {extracted_text}")
        else:
            print("[EasyOCR RESULT]: Koi clear text nahi mila.")
            speak("No text found.")

    # Object Detection (press 'o')
    elif key == ord('o'):
        detected_names = [yolo_model.names[int(box.cls[0])] for box in results[0].boxes]
        unique_objects = list(set(detected_names))
        
        if unique_objects:
            obj_str = ", ".join(unique_objects)
            print(f"[OBJECTS]: {obj_str}")
            speak(f"I see {obj_str}")
        else:
            print("[OBJECTS]: Koi object detect nahi hua.")
            speak("No objects detected.")

    elif key == ord('q'):
        speak("Shutting down.")
        break

cap.release()
cv2.destroyAllWindows()
