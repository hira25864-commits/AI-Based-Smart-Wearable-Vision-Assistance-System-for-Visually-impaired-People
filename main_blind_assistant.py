import cv2
import pytesseract
import pyttsx3
import threading
from ultralytics import YOLO

# Audio Engine Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speaking Speed

def speak(text):
    """Background thread mein audio bolne ke liye function"""
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

print("[INFO] Models aur Audio load ho rahe hain...")
yolo_model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam open nahi ho saka.")
    exit()

speak("System Ready. Press T to read text, O to announce objects.")

print("\n=== CONTROLS ===")
print("Press 't' -> Read Text (OCR + Voice)")
print("Press 'o' -> Announce Detected Objects (YOLO + Voice)")
print("Press 'q' -> Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Real-time YOLO Detection
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Smart Assistance System", annotated_frame)
    key = cv2.waitKey(1) & 0xFF

    # 1. OCR Trigger (press 't')
    if key == ord('t'):
        print("\n[OCR] Text read ho raha hai...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        
        if text:
            print(f"[OCR RESULT]:\n{text}")
            speak(f"Text detected: {text}")
        else:
            print("[OCR RESULT]: Koi clear text nahi mila.")
            speak("No text found.")

    # 2. YOLO Object Announcement (press 'o')
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
