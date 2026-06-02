import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


class GestureDetector:
    GESTURE_LABELS = [
        "Ambas cejas levantadas",
        "Guiñar",
        "Comisura derecha",
        "Comisura izquierda",
        "Morder labio inferior",
        "Beso",
        "Abrir la boca",
    ]

    def __init__(
        self,
        brow_threshold=0.30,
        wink_threshold=0.50,
        wink_other_max=0.50,
        kiss_threshold=0.30,
        jaw_open_threshold=0.20,
        wink_hysteresis=3,
        bite_hysteresis=3,
    ):
        self.brow_threshold = brow_threshold
        self.wink_threshold = wink_threshold
        self.wink_other_max = wink_other_max
        self.kiss_threshold = kiss_threshold
        self.jaw_open_threshold = jaw_open_threshold
        self.wink_hysteresis = wink_hysteresis
        self.bite_hysteresis = bite_hysteresis
        self._wink_counter = 0
        self._bite_counter = 0
        self._bite_active = False
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
            output_face_blendshapes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        h, w = frame.shape[:2]
        active = [False] * len(self.GESTURE_LABELS)
        blendshapes = {}
        landmarks = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = cat.score
            active = self._check_gestures(blendshapes, landmarks)

        self._draw_gestures(frame, active)
        return frame, active

    def _check_gestures(self, bs, landmarks=None):
        active = [False] * len(self.GESTURE_LABELS)

        brow_l = bs.get("browOuterUpLeft", 0)
        brow_r = bs.get("browOuterUpRight", 0)
        active[0] = brow_l > self.brow_threshold and brow_r > self.brow_threshold

        blink_l = bs.get("eyeBlinkLeft", 0)
        blink_r = bs.get("eyeBlinkRight", 0)
        raw_wink = (blink_l > self.wink_threshold and blink_r < self.wink_other_max) or \
                   (blink_r > self.wink_threshold and blink_l < self.wink_other_max)

        if raw_wink:
            self._wink_counter = min(self._wink_counter + 1, self.wink_hysteresis + 1)
        else:
            self._wink_counter = 0
        active[1] = self._wink_counter >= self.wink_hysteresis

        if landmarks is not None:
            der, izq = self._check_mouth_corners(landmarks)
            active[2] = der
            active[3] = izq
        else:
            active[2] = False
            active[3] = False

        jaw_open = bs.get("jawOpen", 0)
        shrug_lower = bs.get("mouthShrugLower", 0)
        raw_bite = jaw_open > 0.006 and jaw_open < 0.04 and shrug_lower < 0.18

        if raw_bite:
            self._bite_counter = min(self._bite_counter + 1, self.bite_hysteresis + 1)
        else:
            self._bite_counter = 0
        self._bite_active = self._bite_counter >= self.bite_hysteresis
        active[4] = self._bite_active

        pucker = bs.get("mouthPucker", 0)
        funnel = bs.get("mouthFunnel", 0)
        active[5] = (pucker + funnel) / 2 > self.kiss_threshold

        active[6] = jaw_open > self.jaw_open_threshold

        return active

    def _check_mouth_corners(self, landmarks):
        c61 = landmarks[61]
        c291 = landmarks[291]
        center_top = landmarks[0]
        center_bottom = landmarks[17]

        center_y = (center_top.y + center_bottom.y) / 2

        raise_61 = max(0, center_y - c61.y)
        raise_291 = max(0, center_y - c291.y)

        nose = landmarks[1]
        chin = landmarks[152]
        face_height = chin.y - nose.y

        norm_61 = raise_61 / face_height
        norm_291 = raise_291 / face_height

        diff = abs(norm_61 - norm_291)
        person_izq = norm_61 > 0.015 and norm_61 > norm_291 * 1.8 and diff > 0.01
        person_der = norm_291 > 0.015 and norm_291 > norm_61 * 1.8 and diff > 0.01

        return person_der, person_izq

    def _draw_gestures(self, frame, active):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 280), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.putText(
            frame, "GESTOS DETECTADOS", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        for i, label in enumerate(self.GESTURE_LABELS):
            y = 65 + i * 30
            if active[i]:
                cv2.putText(
                    frame, f"  {label}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    frame, f"  {label}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1,
                )
