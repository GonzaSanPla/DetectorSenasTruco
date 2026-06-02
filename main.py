import cv2
import sys
from gesture_detector import GestureDetector

CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    detector = GestureDetector()
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camara {CAM_INDEX} | Resolucion real: {real_w}x{real_h}  Presiona 'q' para salir")

    cv2.namedWindow("Detector de Senas - Truco", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame")
            break

        frame = cv2.flip(frame, 1)

        frame, active = detector.process_frame(frame)

        cv2.imshow("Detector de Senas - Truco", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Detector de Senas - Truco", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
