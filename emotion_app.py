from collections import deque
from pathlib import Path
import sys

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import cv2

from hand_tracker import HandTracker

try:
    from fer import FER
except ImportError:
    try:
        from fer.fer import FER
    except Exception as exc:
        raise SystemExit(
            "FER could not be imported. Install the required packages first, including "
            "'tensorflow-cpu', then run this script again.\n"
            f"Import error: {exc}"
        )
except Exception as exc:
    raise SystemExit(
        "FER could not be imported. Install the required packages first, including "
        "'tensorflow-cpu', then run this script again.\n"
        f"Import error: {exc}"
    )


class MoodGestureApp:
    """Real-time AI mood detection using FER + MediaPipe hand tracking."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.hand_tracker = HandTracker(max_num_hands=1)
        self.emotion_detector = FER(mtcnn=False)
        self.supported_emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]
        self.emotion_score_history = deque(maxlen=5)

    def run(self):
        camera = cv2.VideoCapture(self.camera_index)

        if not camera.isOpened():
            raise SystemExit(
                "Could not access the webcam. Make sure your camera is connected "
                "and not being used by another application."
            )

        print("Starting AI Gesture & Emotion Recognition System...")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)

            hand_data = self.hand_tracker.process_frame(frame)
            emotion_result = self.detect_main_emotion(frame)

            self.hand_tracker.draw_hands(frame, hand_data)
            self.draw_emotion_box(frame, emotion_result)
            self.draw_dashboard(frame, emotion_result, hand_data)

            cv2.imshow("AI Gesture & Emotion Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        camera.release()
        self.hand_tracker.close()
        cv2.destroyAllWindows()

    def detect_main_emotion(self, frame):
        detections = self.emotion_detector.detect_emotions(frame)

        if not detections:
            self.emotion_score_history.clear()
            return {
                "emotion": "No face detected",
                "score": 0.0,
                "box": None,
                "scores": {},
            }

        main_face = max(
            detections,
            key=lambda item: item["box"][2] * item["box"][3],
        )
        current_scores = self.normalize_scores(main_face["emotions"])
        box = self.normalize_box(main_face["box"])
        self.emotion_score_history.append(current_scores)
        smoothed_scores = self.get_smoothed_scores()
        stable_emotion = max(smoothed_scores, key=smoothed_scores.get)
        stable_score = smoothed_scores[stable_emotion]

        return {
            "emotion": stable_emotion,
            "score": stable_score,
            "box": box,
            "scores": smoothed_scores,
        }

    def get_smoothed_scores(self):
        if not self.emotion_score_history:
            return {emotion: 0.0 for emotion in self.supported_emotions}

        smoothed_scores = {}
        history_length = len(self.emotion_score_history)
        for emotion in self.supported_emotions:
            total = sum(frame_scores.get(emotion, 0.0) for frame_scores in self.emotion_score_history)
            smoothed_scores[emotion] = total / history_length

        return smoothed_scores

    def draw_emotion_box(self, frame, emotion_result):
        if emotion_result["box"] is None:
            return

        x_pos, y_pos, width, height = emotion_result["box"]
        emotion_text = (
            f'{emotion_result["emotion"].capitalize()} '
            f'({emotion_result["score"]:.2f})'
        )

        cv2.rectangle(
            frame,
            (x_pos, y_pos),
            (x_pos + width, y_pos + height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            emotion_text,
            (x_pos, max(25, y_pos - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def normalize_scores(self, scores):
        normalized_scores = {}
        for emotion in self.supported_emotions:
            normalized_scores[emotion] = float(scores.get(emotion, 0.0))

        return normalized_scores

    def normalize_box(self, box):
        """Convert FER box output into a simple (x, y, w, h) tuple."""
        if box is None:
            return None

        if hasattr(box, "tolist"):
            box = box.tolist()

        if len(box) != 4:
            return None

        return tuple(int(value) for value in box)

    def draw_dashboard(self, frame, emotion_result, hand_data):
        emotion = emotion_result["emotion"]
        primary_hand = hand_data[0] if hand_data else None
        finger_count = primary_hand["finger_count"] if primary_hand else 0
        gesture = primary_hand["gesture"] if primary_hand else "No hand detected"
        response = self.generate_smart_response(emotion, finger_count, gesture)
        top_emotions = self.get_top_emotions(emotion_result.get("scores", {}), limit=4)

        panel_height = 250
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        lines = [
            f"Emotion: {emotion.capitalize() if emotion != 'No face detected' else emotion}",
            f"Finger Count: {finger_count}",
            f"Gesture: {gesture}",
            f"Response: {response}",
        ]

        y_pos = 30
        for line in lines:
            cv2.putText(
                frame,
                line,
                (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y_pos += 30

        if top_emotions:
            cv2.putText(
                frame,
                "Top Mood Scores:",
                (15, y_pos + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            y_pos += 35

            for mood_name, mood_score in top_emotions:
                cv2.putText(
                    frame,
                    f"{mood_name.capitalize()}: {mood_score:.2f}",
                    (25, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    2,
                )
                y_pos += 25

    def get_top_emotions(self, scores, limit=4):
        if not scores:
            return []

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores[:limit]

    def generate_smart_response(self, emotion, finger_count, gesture):
        emotion = emotion.lower()
        gesture = gesture.lower()

        if emotion == "happy" and finger_count == 5:
            return "Awesome!"
        if emotion == "happy":
            return "You look cheerful."
        if emotion == "sad" and gesture == "closed fist":
            return "Stay strong."
        if emotion == "sad":
            return "Take a deep breath."
        if emotion == "angry":
            return "Relax and breathe."
        if emotion == "surprise":
            return "That is unexpected!"
        if emotion == "fear":
            return "You are safe. Take it slow."
        if emotion == "disgust":
            return "Something feels off."
        if emotion == "neutral" and gesture == "victory":
            return "Calm and confident."
        if emotion == "neutral":
            return "Steady and focused."
        if gesture == "thumbs up":
            return "Nice gesture!"
        if gesture == "open palm":
            return "Great energy."
        if emotion == "no face detected":
            return "Face not visible yet."
        return "Tracking mood and gesture in real time."


def main():
    app = MoodGestureApp(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
