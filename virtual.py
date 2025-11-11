import cv2
import numpy as np
import time
from HandTrackingModule import handDetector  
from tensorflow.keras.models import load_model

# =========================
# Load EMNIST model
# =========================
model = load_model('E:\\virtual_world\\emnist_letters_model.keras')  
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# =========================
# Helper: Convert path to 28x28 image for model input
# =========================
def path_to_image(canvas, path):
    if len(path) < 2:
        return None

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    min_x, max_x = max(min(xs)-10, 0), min(max(xs)+10, canvas.shape[1])
    min_y, max_y = max(min(ys)-10, 0), min(max(ys)+10, canvas.shape[0])

    cropped = canvas[min_y:max_y, min_x:max_x]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    img28 = cv2.resize(thresh, (28, 28))
    img28 = 255 - img28  # invert to match EMNIST
    img28 = img28.astype(np.float32) / 255.0
    img28 = np.expand_dims(img28, axis=(0, -1))  # shape: (1,28,28,1)
    return img28

# =========================
# Hand Letter Recognizer
# =========================
class HandLetterRecognizer:
    def __init__(self, model):
        self.drawing = False
        self.path = []
        self.last_seen = 0
        self.time_limit = 1.5  # seconds to finish drawing
        self.model = model
        self.letters = letters
        self.imgCanvas = None
        self.last_prediction = ""  # store last predicted letter & confidence

    def start_drawing(self):
        self.drawing = True
        self.path = []

    def recognize_letter(self):
        img = path_to_image(self.imgCanvas, self.path)
        if img is None:
            return None, 0.0
        preds = self.model.predict(img, verbose=0)
        idx = np.argmax(preds)
        confidence = float(preds[0][idx])
        if confidence > 0.5:
            return self.letters[idx], confidence
        return None, confidence

    def update(self, frame, detector):
        lmList = detector.findPosition(frame, draw=False)
        if len(lmList) > 0:
            fingers = detector.fingersUp()
            index_tip = lmList[8][1], lmList[8][2]

            # Drawing mode: index finger up only
            if fingers[1] == 1 and fingers[2] == 0:
                if not self.drawing:
                    self.start_drawing()
                    self.imgCanvas = np.zeros_like(frame)
                self.path.append(index_tip)
                # Draw on canvas
                if len(self.path) > 1:
                    for i in range(len(self.path)-1):
                        cv2.line(self.imgCanvas, self.path[i], self.path[i+1], (0,0,255), 8)
                self.last_seen = time.time()

            # Finish drawing if hand removed for > time_limit
            elif self.drawing and (time.time() - self.last_seen > self.time_limit):
                self.drawing = False
                letter, conf = self.recognize_letter()
                self.path = []
                self.imgCanvas = None
                if letter:
                    self.last_prediction = f"{letter} ({conf*100:.1f}%)"
                    return letter
        return None

# =========================
# Main App
# =========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Set camera resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # height
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

    detector = handDetector(detectionCon=0.75)
    recognizer = HandLetterRecognizer(model)
    typed_text = ""

    print(" Starting Virtual Letter Recognizer... Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)

        letter = recognizer.update(frame, detector)
        if letter:
            print(f"Recognized: {letter}")
            typed_text += letter

        # Overlay canvas
        if recognizer.imgCanvas is not None:
            frame = cv2.addWeighted(frame, 0.5, recognizer.imgCanvas, 0.5, 0)

        # Show last prediction confidence on screen
        if recognizer.last_prediction:
            cv2.putText(frame, "Last: " + recognizer.last_prediction, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display typed text
        cv2.putText(frame, "Typed: " + typed_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Hand Letter Drawing", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 8:  # Backspace
            typed_text = typed_text[:-1]
        elif key == ord(' '):  # Space
            typed_text += ' '

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
