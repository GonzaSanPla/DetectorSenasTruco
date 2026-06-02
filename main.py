import cv2
from eyebrow_detector import EyebrowDetector


def main():
    detector = EyebrowDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam")
        return

    print("Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame")
            break

        frame = cv2.flip(frame, 1)

        frame, state = detector.process_frame(frame)

        cv2.imshow("Detector de Cejas - Truco", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
