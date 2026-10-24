import cv2
import easyocr
import pyttsx3
import os

# Audio Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print(f"[VOICE]: {text}")
    engine.say(text)
    engine.runAndWait()

# Sample image download
print("[INFO] Sample test image download ho rahi hai...")
os.system("wget -O sample.jpg https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png")

print("[INFO] EasyOCR reader load ho raha hai...")
reader = easyocr.Reader(['en'], gpu=False)

image_path = 'sample.jpg'
if os.path.exists(image_path):
    print("[INFO] Image processing shuru...")
    speak("Reading text from image.")
    
    ocr_results = reader.readtext(image_path)
    extracted_text = " ".join([text[1] for text in ocr_results]).strip()
    
    print("\n" + "=" * 40)
    print(f"[EasyOCR RESULT]:\n{extracted_text}")
    print("=" * 40 + "\n")
    
    if extracted_text:
        speak(f"Text detected: {extracted_text}")
    else:
        speak("No text found.")
else:
    print("[ERROR] Sample image nahi mili.")
