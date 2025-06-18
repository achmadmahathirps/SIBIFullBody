import math
import time
from collections import deque
import keyboard
import cv2 as opencv
import mediapipe

# Global variables
fps_buffer = deque(maxlen=10)


def update_fps(previous_time):
    current_time = time.time()
    delta = current_time - previous_time
    fps = 1/delta if delta != 0 else 0

    fps_buffer.append(fps)
    smoothed_fps = sum(fps_buffer) / len(fps_buffer)

    return smoothed_fps, current_time


def euclidean_distance(point1, point2):
    return math.hypot(point1.x - point2.y, point1.y - point2.y)


def normalize_landmarks(landmarks, center, scale):
    return [((lm.x - center[0]) / scale, (lm.y - center[1]) / scale) for lm in landmarks]


def draw_fps(frame, fps, position=(10, 30), scale=1, thickness=2):
    if fps < 10:
        color = (0, 0, 255) # RED (BGR)
    elif fps < 20:
        color = (0, 165, 255) # Orange (BGR)
    else:
        color = (0, 255, 0) # Green (BGR)

    opencv.putText(
        frame,
        f'FPS: {int(fps)}',
        position,
        opencv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        opencv.LINE_AA,
    )


def main():

    # Initialize webcam settings
    webcam = int(1)
    from_capture = opencv.VideoCapture(webcam, opencv.CAP_DSHOW)
    from_capture.set(opencv.CAP_PROP_FRAME_WIDTH, 640)
    from_capture.set(opencv.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize mediapipe detection settings
    mediapipe_holistic = mediapipe.solutions.holistic
    mediapipe_drawing = mediapipe.solutions.drawing_utils
    mediapipe_drawing_styles = mediapipe.solutions.drawing_styles

    # Initialize fps settings
    previous_time = time.time()

    # If the camera is not detected
    if not from_capture.isOpened():
        print("Cannot open camera")
        exit()

    with mediapipe_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        # While the camera is detected
        while True:

            # Capture frame-by-frame from the camera
            available, frame = from_capture.read()

            # If there is no available frame left or is not detected
            if not available:
                print("No frame left (stream end?). Exiting...")
                break

            # Flip the image horizontally
            frame = opencv.flip(frame, 1)

            # Optimize before detection process
            frame.flags.writeable = False
            frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
            detection_results = holistic.process(frame)
            frame.flags.writeable = True
            frame = opencv.cvtColor(frame, opencv.COLOR_RGB2BGR)

            if detection_results.pose_landmarks is not None:
                # 

                # Draw pose landmarks
                mediapipe_drawing.draw_landmarks(
                    frame,
                    detection_results.pose_landmarks,
                    mediapipe_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec = mediapipe_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Draw right hand landmarks
                mediapipe_drawing.draw_landmarks(
                    frame,
                    detection_results.right_hand_landmarks,
                    mediapipe_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec = mediapipe_drawing_styles.get_default_hand_landmarks_style(),
                )

                # Draw left hand landmarks
                mediapipe_drawing.draw_landmarks(
                    frame,
                    detection_results.left_hand_landmarks,
                    mediapipe_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec = mediapipe_drawing_styles.get_default_hand_landmarks_style(),
                )
            else:
                pass

            # Update & draw FPS
            fps, previous_time = update_fps(previous_time)
            draw_fps(frame, fps)

            # Show the results in the window
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