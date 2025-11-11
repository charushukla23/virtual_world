import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from collections import deque

# Optional: brightness control on Windows/Linux
try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

# --- CONFIGURATION ---
class Config:
    """Configuration settings for the application."""
    # General
    CAM_INDEX = 0  # 0 for default camera
    FLIP_CAM = True
    SCREEN_W, SCREEN_H = pyautogui.size()

    # Smoothing and Thresholds
    CURSOR_SMOOTHING = 5
    BLINK_COOLDOWN_S = 1.2
    CLICK_THRESH_PX = 35
    SCROLL_ACTIVATION_PX = 120
    ACTION_COOLDOWN_S = 0.4

    # Control Sensitivity
    SCROLL_SENSITIVITY = 1.5
    VOL_BRIGHT_SENSITIVITY = 40
    VOL_BRIGHT_STEP = 5

# --- MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class EyeTracker:
    """Handles eye tracking for gaze estimation and blink detection."""
    def __init__(self):
        self.last_blink_time = 0
        self.blink_registered = False
        self.LEFT_EYE_TOP_BOTTOM = [386, 374]
        self.RIGHT_EYE_TOP_BOTTOM = [159, 145]
        self.LEFT_EYE_IRIS_CENTER = 474

    def _calculate_ear(self, landmarks, eye_indices, img_h):
        top_pt = landmarks[eye_indices[0]]
        bottom_pt = landmarks[eye_indices[1]]
        return abs(int(top_pt.y * img_h) - int(bottom_pt.y * img_h))

    def process_face(self, frame, ui):
        img_h, img_w, _ = frame.shape
        face_results = face_mesh.process(frame)
        gaze_pt = None
        self.blink_registered = False
        if face_results.multi_face_landmarks:
            mesh = face_results.multi_face_landmarks[0].landmark
            iris = mesh[self.LEFT_EYE_IRIS_CENTER]
            gaze_x = int(iris.x * ui.width)
            gaze_y = int(iris.y * ui.height)
            gaze_pt = (gaze_x, gaze_y)
            ear_left = self._calculate_ear(mesh, self.LEFT_EYE_TOP_BOTTOM, img_h)
            ear_right = self._calculate_ear(mesh, self.RIGHT_EYE_TOP_BOTTOM, img_h)
            blink_threshold = (ui.key_size[1] / 7)
            if ear_left < blink_threshold and ear_right < blink_threshold:
                if time.time() - self.last_blink_time > Config.BLINK_COOLDOWN_S:
                    self.last_blink_time = time.time()
                    self.blink_registered = True
        return gaze_pt

class GestureController:
    """Handles hand gesture detection and translation into system commands."""
    def __init__(self):
        self.cursor_hist = deque(maxlen=Config.CURSOR_SMOOTHING)
        self.last_click_time = 0
        self.active_mode = "None"
        self.mode_origin_y = None
        self.IDX_TIP, self.MID_TIP, self.RING_TIP, self.PINKY_TIP, self.THUMB_TIP = 8, 12, 16, 20, 4

    def _get_landmark_coords(self, landmarks, img_w, img_h):
        return {i: (int(p.x * img_w), int(p.y * img_h)) for i, p in enumerate(landmarks)}

    def _adjust_system_control(self, control_type, delta):
        if control_type == "Volume":
            pyautogui.press('volumeup' if delta > 0 else 'volumedown')
        elif control_type == "Brightness" and sbc:
            try:
                current = sbc.get_brightness()[0]
                new = np.clip(current + (Config.VOL_BRIGHT_STEP if delta > 0 else -Config.VOL_BRIGHT_STEP), 0, 100)
                sbc.set_brightness(new)
            except Exception: pass

    def process_hand(self, frame):
        hand_results = hands.process(frame)
        if not hand_results.multi_hand_landmarks:
            if self.active_mode == "Drag": pyautogui.mouseUp()
            self.active_mode = "None"
            return
        landmarks = hand_results.multi_hand_landmarks[0].landmark
        pts = self._get_landmark_coords(landmarks, Config.SCREEN_W, Config.SCREEN_H)
        ix, iy = pts[self.IDX_TIP]
        self.cursor_hist.append((ix, iy))
        sm_x, sm_y = np.mean(self.cursor_hist, axis=0, dtype=int)
        pyautogui.moveTo(sm_x, sm_y, duration=0)
        d_idx_mid = np.hypot(pts[self.IDX_TIP][0] - pts[self.MID_TIP][0], pts[self.IDX_TIP][1] - pts[self.MID_TIP][1])
        d_idx_thumb = np.hypot(ix - pts[self.THUMB_TIP][0], iy - pts[self.THUMB_TIP][1])
        d_mid_ring = np.hypot(pts[self.MID_TIP][0] - pts[self.RING_TIP][0], pts[self.MID_TIP][1] - pts[self.RING_TIP][1])
        d_ring_thumb = np.hypot(pts[self.RING_TIP][0] - pts[self.THUMB_TIP][0], pts[self.RING_TIP][1] - pts[self.THUMB_TIP][1])
        d_pinky_thumb = np.hypot(pts[self.PINKY_TIP][0] - pts[self.THUMB_TIP][0], pts[self.PINKY_TIP][1] - pts[self.THUMB_TIP][1])
        now = time.time()
        if self.active_mode != "Drag": self.active_mode = "None"
        if d_idx_mid < Config.CLICK_THRESH_PX and now - self.last_click_time > Config.ACTION_COOLDOWN_S:
            pyautogui.click(); self.last_click_time = now
        elif d_mid_ring < Config.CLICK_THRESH_PX and now - self.last_click_time > Config.ACTION_COOLDOWN_S:
            pyautogui.rightClick(); self.last_click_time = now
        elif d_idx_thumb < Config.CLICK_THRESH_PX:
            if self.active_mode != "Drag": pyautogui.mouseDown(); self.active_mode = "Drag"
        elif self.active_mode == "Drag": pyautogui.mouseUp(); self.active_mode = "None"
        elif d_pinky_thumb < Config.CLICK_THRESH_PX:
            self.active_mode = "Volume"
            if self.mode_origin_y is None: self.mode_origin_y = pts[self.PINKY_TIP][1]
            delta = self.mode_origin_y - pts[self.PINKY_TIP][1]
            if abs(delta) > Config.VOL_BRIGHT_SENSITIVITY: self._adjust_system_control("Volume", delta); self.mode_origin_y = pts[self.PINKY_TIP][1]
        elif d_ring_thumb < Config.CLICK_THRESH_PX:
            self.active_mode = "Brightness"
            if self.mode_origin_y is None: self.mode_origin_y = pts[self.RING_TIP][1]
            delta = self.mode_origin_y - pts[self.RING_TIP][1]
            if abs(delta) > Config.VOL_BRIGHT_SENSITIVITY: self._adjust_system_control("Brightness", delta); self.mode_origin_y = pts[self.RING_TIP][1]
        elif d_idx_mid > Config.SCROLL_ACTIVATION_PX:
            self.active_mode = "Scroll"
            if self.mode_origin_y is None: self.mode_origin_y = iy
            pyautogui.scroll(int(-(iy - self.mode_origin_y) / Config.SCROLL_SENSITIVITY))
        if self.active_mode not in ["Volume", "Brightness", "Scroll"]: self.mode_origin_y = None

class UI:
    """Manages the drawing of the user interface, including keyboard and status."""
    def __init__(self):
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<-'],
            ['SPACE']
        ]
        self.key_size = (80, 80)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.margin = 20
        self.kb_start_y = 200
        self.width = max(len(row) for row in self.keys) * self.key_size[0] + self.margin * 2
        self.height = self.kb_start_y + len(self.keys) * self.key_size[1] + self.margin
        self.typed_text = ""
        self.last_typed_time = 0

    def _draw_text(self, img, text, pos, scale=1.0, color=(0, 0, 0), thickness=2, center=False):
        if center:
            text_size = cv2.getTextSize(text, self.font, scale, thickness)[0]
            pos = (pos[0] - text_size[0] // 2, pos[1] + text_size[1] // 2)
        cv2.putText(img, text, pos, self.font, scale, color, thickness, cv2.LINE_AA)

    def draw_keyboard(self, canvas, gaze_pt, blink_occurred):
        for r, row in enumerate(self.keys):
            row_width = len(row) * self.key_size[0]
            start_x = (self.width - row_width) // 2
            if row[0] == 'SPACE': # Center the spacebar row
                start_x = (self.width - self.key_size[0] * len(self.keys[1])) // 2 # Align with row above
            for c, key in enumerate(row):
                key_w = self.key_size[0]
                # Dynamic width for SPACE key, making it span the width of the row above it
                if key == 'SPACE':
                    key_w = len(self.keys[1]) * self.key_size[0] # Match width of the 'ASDF' row
                x = start_x + c * self.key_size[0]
                y = self.kb_start_y + r * self.key_size[1]
                bg_color = (200, 200, 200)
                if gaze_pt and x < gaze_pt[0] < x + key_w and y < gaze_pt[1] < y + self.key_size[1]:
                    bg_color = (150, 255, 150)
                    if blink_occurred and time.time() - self.last_typed_time > 0.5:
                        bg_color = (0, 255, 0)
                        if key == '<-': self.typed_text = self.typed_text[:-1]
                        elif key == 'SPACE': self.typed_text += ' '
                        else: self.typed_text += key
                        self.last_typed_time = time.time()
                cv2.rectangle(canvas, (x, y), (x + key_w, y + self.key_size[1]), bg_color, -1)
                cv2.rectangle(canvas, (x, y), (x + key_w, y + self.key_size[1]), (50, 50, 50), 2)
                self._draw_text(canvas, key, (x + key_w//2, y + self.key_size[1]//2), 1.2, center=True)

    def draw_main_canvas(self, cam_frame, gaze_pt, gesture_mode, eye_tracker):
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        cam_h, cam_w, _ = cam_frame.shape
        canvas[self.margin : self.margin + cam_h, self.margin : self.margin + cam_w] = cam_frame
        self._draw_text(canvas, f"Typed: {self.typed_text}", (cam_w + self.margin * 2, self.margin + 40), 1.0, (50, 50, 50))
        self._draw_text(canvas, f"Mode: {gesture_mode}", (cam_w + self.margin * 2, self.margin + 90), 1.0, (200, 50, 50))
        self.draw_keyboard(canvas, gaze_pt, eye_tracker.blink_registered)
        if gaze_pt:
            cv2.circle(canvas, gaze_pt, 10, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, gaze_pt, 2, (255, 0, 0), -1, cv2.LINE_AA)
        return canvas

    def create_gesture_guide_image(self):
        """Creates a separate image for the gesture guide."""
        guide_h, guide_w = 200, 450
        guide_img = np.ones((guide_h, guide_w, 3), dtype=np.uint8) * 240
        self._draw_text(guide_img, "Gesture Guide", (guide_w // 2, 25), 0.8, color=(0,0,0), thickness=2, center=True)
        cv2.line(guide_img, (20, 40), (guide_w - 20, 40), (50,50,50), 1)
        gestures = [
            ("Left Click:", "Index + Middle Pinch"),
            ("Right Click:", "Middle + Ring Pinch"),
            ("Drag:", "Index + Thumb (Hold)"),
            ("Scroll:", "Index/Middle Apart + Move"),
            ("Volume:", "Pinky + Thumb + Move"),
            ("Brightness:", "Ring + Thumb + Move"),
            ("Type:", "Gaze at Key + Blink")
        ]
        for i, (action, gesture) in enumerate(gestures):
            y_pos = 65 + i * 20
            self._draw_text(guide_img, action, (30, y_pos), 0.6, (150, 50, 50), 2)
            self._draw_text(guide_img, gesture, (200, y_pos), 0.6, (50, 50, 50), 2)
        return guide_img

def main():
    """Main function to run the application."""
    cap = cv2.VideoCapture(Config.CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {Config.CAM_INDEX}.")
        return
    

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    
    ui = UI()
    eye_tracker = EyeTracker()
    gesture_controller = GestureController()

    # Create and display the persistent gesture guide window
    gesture_guide_img = ui.create_gesture_guide_image()
    cv2.imshow("Gesture Guide", gesture_guide_img)

    while True:
        ok, frame = cap.read()
        if not ok: break
        if Config.FLIP_CAM: frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gaze_pt = eye_tracker.process_face(rgb_frame, ui)
        gesture_controller.process_hand(rgb_frame)

        small_cam = cv2.resize(frame, (240, 180))
        canvas = ui.draw_main_canvas(small_cam, gaze_pt, gesture_controller.active_mode, eye_tracker)

        cv2.imshow("Eye & Hand Control", canvas)
        # Allow the guide window to be moved and closed
        cv2.waitKey(1)
        if cv2.getWindowProperty("Eye & Hand Control", cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty("Gesture Guide", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()