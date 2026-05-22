from pathlib import Path
import sys
import time
import urllib.request

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import cv2
import mediapipe as mp


class HandTracker:
    """Detect hands, count fingers, and recognize a few simple gestures."""

    TIP_IDS = [4, 8, 12, 16, 20]
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )

    def __init__(
        self,
        max_num_hands=1,
        detection_confidence=0.6,
        tracking_confidence=0.6,
        model_path=None,
    ):
        self.max_num_hands = max_num_hands
        self.model_path = Path(model_path) if model_path else self._default_model_path()
        self._ensure_model_exists()

        self.base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        self.vision = mp.tasks.vision
        self.drawing_utils = self.vision.drawing_utils
        self.drawing_styles = self.vision.drawing_styles
        self.hand_connections = self.vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.last_timestamp_ms = 0

        options = self.vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=self.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.hand_landmarker = self.vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """
        Process a BGR frame and return hand information.

        Each hand dictionary contains:
        - label: Left or Right
        - landmarks: pixel coordinates for 21 hand landmarks
        - normalized_landmarks: normalized MediaPipe landmarks
        - finger_states: list of 5 integers (0 or 1)
        - finger_count: total raised fingers
        - gesture: basic gesture name
        - bbox: bounding box for UI labels
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = self._next_timestamp_ms()
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_data = []
        if not result.hand_landmarks:
            return hand_data

        frame_height, frame_width, _ = frame.shape

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            pixel_landmarks = self._extract_landmarks(
                landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            label = handedness[0].category_name if handedness else "Unknown"
            finger_states = self._get_finger_states(pixel_landmarks, label)
            finger_count = sum(finger_states)
            gesture = self._recognize_gesture(finger_states)
            bbox = self._get_bounding_box(pixel_landmarks)

            hand_data.append(
                {
                    "label": label,
                    "landmarks": pixel_landmarks,
                    "normalized_landmarks": landmarks,
                    "finger_states": finger_states,
                    "finger_count": finger_count,
                    "gesture": gesture,
                    "bbox": bbox,
                }
            )

        return hand_data

    def draw_hands(self, frame, hand_data):
        """Draw landmarks, connections, and labels on the frame."""
        for hand in hand_data:
            self.drawing_utils.draw_landmarks(
                frame,
                hand["normalized_landmarks"],
                self.hand_connections,
                self.drawing_styles.get_default_hand_landmarks_style(),
                self.drawing_styles.get_default_hand_connections_style(),
            )

            x_pos, y_pos, _, _ = hand["bbox"]
            label_text = (
                f'{hand["label"]}: {hand["finger_count"]} finger(s) | {hand["gesture"]}'
            )
            cv2.putText(
                frame,
                label_text,
                (x_pos, max(25, y_pos - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        return frame

    def close(self):
        self.hand_landmarker.close()

    def _default_model_path(self):
        return Path(__file__).resolve().parent / ".models" / "hand_landmarker.task"

    def _ensure_model_exists(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists():
            return

        try:
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the MediaPipe hand model automatically. "
                f"Download it manually from {self.MODEL_URL} and save it to "
                f"{self.model_path}.\nOriginal error: {exc}"
            ) from exc

    def _next_timestamp_ms(self):
        current_time_ms = int(time.time() * 1000)
        if current_time_ms <= self.last_timestamp_ms:
            current_time_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = current_time_ms
        return current_time_ms

    def _extract_landmarks(self, landmarks, frame_width, frame_height):
        pixel_landmarks = []

        for landmark in landmarks:
            x_coord = int(landmark.x * frame_width)
            y_coord = int(landmark.y * frame_height)
            pixel_landmarks.append((x_coord, y_coord))

        return pixel_landmarks

    def _get_finger_states(self, landmarks, label):
        fingers = []

        thumb_tip_x = landmarks[self.TIP_IDS[0]][0]
        thumb_joint_x = landmarks[self.TIP_IDS[0] - 1][0]
        if label == "Right":
            fingers.append(1 if thumb_tip_x < thumb_joint_x else 0)
        else:
            fingers.append(1 if thumb_tip_x > thumb_joint_x else 0)

        for tip_id in self.TIP_IDS[1:]:
            fingertip_y = landmarks[tip_id][1]
            upper_joint_y = landmarks[tip_id - 2][1]
            fingers.append(1 if fingertip_y < upper_joint_y else 0)

        return fingers

    def _recognize_gesture(self, finger_states):
        if finger_states == [0, 0, 0, 0, 0]:
            return "Closed Fist"
        if finger_states == [1, 1, 1, 1, 1]:
            return "Open Palm"
        if finger_states == [0, 1, 1, 0, 0]:
            return "Victory"
        if finger_states == [1, 0, 0, 0, 0]:
            return "Thumbs Up"
        if finger_states == [0, 1, 0, 0, 0]:
            return "Pointing"
        return "Custom Gesture"

    def _get_bounding_box(self, landmarks):
        x_values = [point[0] for point in landmarks]
        y_values = [point[1] for point in landmarks]
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)
        return x_min, y_min, x_max - x_min, y_max - y_min
