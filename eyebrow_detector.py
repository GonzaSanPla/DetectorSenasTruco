import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


class EyebrowDetector:
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    LEFT_UPPER_EYELID = 159
    RIGHT_UPPER_EYELID = 386
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    STATES = [
        "Ninguna levantada",
        "Izquierda levantada",
        "Derecha levantada",
        "Ambas levantadas",
    ]

    def __init__(self, threshold=0.30):
        self.threshold = threshold
        self._ensure_model()
        self._init_landmarker()

    def _ensure_model(self):
        if not os.path.exists(MODEL_PATH):
            print("Descargando modelo face_landmarker.task ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Modelo descargado")

    def _init_landmarker(self):
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        state = 0
        h, w = frame.shape[:2]

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            left_raised, right_raised = self._check_eyebrows(landmarks, w, h)

            if left_raised:
                state = 1
            if right_raised:
                state = 2
            if left_raised and right_raised:
                state = 3

            self._draw_landmarks(frame, landmarks, w, h)

        self._draw_status(frame, state)
        return frame, state

    def _check_eyebrows(self, landmarks, w, h):
        def avg_y(indices):
            return sum(landmarks[i].y for i in indices) / len(indices)

        left_eyebrow_y = avg_y(self.LEFT_EYEBROW)
        right_eyebrow_y = avg_y(self.RIGHT_EYEBROW)
        left_eyelid_y = landmarks[self.LEFT_UPPER_EYELID].y
        right_eyelid_y = landmarks[self.RIGHT_UPPER_EYELID].y

        left_eye = landmarks[self.LEFT_EYE_OUTER]
        right_eye = landmarks[self.RIGHT_EYE_OUTER]
        inter_eye_dist = np.sqrt(
            (left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2
        )

        left_dist = (left_eyelid_y - left_eyebrow_y) / inter_eye_dist
        right_dist = (right_eyelid_y - right_eyebrow_y) / inter_eye_dist

        return left_dist > self.threshold, right_dist > self.threshold

    def _draw_landmarks(self, frame, landmarks, w, h):
        for idx in self.LEFT_EYEBROW + self.RIGHT_EYEBROW:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def _draw_status(self, frame, state):
        text = self.STATES[state]
        color = (0, 255, 0) if state in (1, 2, 3) else (0, 0, 255)
        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
