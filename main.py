import cv2
from gesture_detector import GestureDetector

CAM_WIDTH = 640
CAM_HEIGHT = 480


def main():
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam")
        return

    print(f"Resolucion: {CAM_WIDTH}x{CAM_HEIGHT}  Presiona 'q' para salir")

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
