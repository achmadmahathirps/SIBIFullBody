import time

import keyboard
import cv2 as opencv
import mediapipe

def main():

    # Initialize webcam settings
    webcam = int(1)
    from_capture = opencv.VideoCapture(webcam, opencv.CAP_DSHOW)
    from_capture.set(opencv.CAP_PROP_FRAME_WIDTH, 640)
    from_capture.set(opencv.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize mediapipe settings
    mediapipe_holistic = mediapipe.solutions.holistic
    holistic = mediapipe_holistic.Holistic(
        static_image_mode = False,
        model_complexity = 1,
        smooth_landmarks = True,
        enable_segmentation = False,
        smooth_segmentation = True,
        refine_face_landmarks = False,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    
    mediapipe_drawing = mediapipe.solutions.drawing_utils
    mediapipe_drawing_styles = mediapipe.solutions.drawing_styles

    # If the camera is not detected
    if not from_capture.isOpened():
        print("Cannot open camera")
        exit()
    # While the camera is detected
    while True:

        # Capture frame-by-frame from the camera
        available, frame = from_capture.read()

        # If there is no available frame left or is not detected
        if not available:
            print("No frame left (stream end?). Exiting...")
            break

        # Mediapipe detection start ------------------------------------------------------------------------------------
        frame.flags.writeable = False
        frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
        detection_results = holistic.process(frame)

        frame.flags.writeable = True
        frame = opencv.cvtColor(frame, opencv.COLOR_RGB2BGR)

        # Draw pose landmark
        mediapipe_drawing.draw_landmarks(
            frame,
            detection_results.pose_landmarks,
            mediapipe_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec = mediapipe_drawing_styles.get_default_pose_landmarks_style(),
        )

        # Draw right hand landmark
        mediapipe_drawing.draw_landmarks(
            frame,
            detection_results.right_hand_landmarks,
            mediapipe_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec = mediapipe_drawing_styles.get_default_hand_landmarks_style(),
        )

        mediapipe_drawing.draw_landmarks(
            frame,
            detection_results.left_hand_landmarks,
            mediapipe_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec = mediapipe_drawing_styles.get_default_hand_landmarks_style(),
        )

        opencv.imshow("SIBI Full-body Edition", frame)

        # Program stops when "ESC" key is pressed
        if opencv.waitKey(3) & keyboard.is_pressed("ESC"):
            print(' ')
            print("(!) Exited through ESC key.")
            break

    from_capture.release()
    opencv.destroyAllWindows()

if __name__ == '__main__':
    main()