import cv2
import cv2 as opencv

def main():

    # Initialize webcam settings
    webcam = int(0)
    from_capture = opencv.VideoCapture(webcam, opencv.CAP_DSHOW)
    from_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    from_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # If the camera is not detected
    if not from_capture.isOpened():
        print("Cannot open camera")
        exit()
    # While the camera is detected
    while True:
        available, frame = from_capture.read()

        if not available:
            print("No frame available")
            break

        opencv.imshow("SIBI Full-body Edition", frame)
        if opencv.waitKey(1) == ord('q'):
            break

    from_capture.release()
    opencv.destroyAllWindows()

if __name__ == '__main__':
    main()