import json
import unittest
from unittest.mock import patch

from EmotionDetection import emotion_detector


class MockResponse:
    def __init__(self, payload_text):
        self.text = payload_text


class EmotionDetectionTests(unittest.TestCase):
    def _run_case(self, statement, expected_dominant, scores):
        payload = {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": scores["anger"],
                        "disgust": scores["disgust"],
                        "fear": scores["fear"],
                        "joy": scores["joy"],
                        "sadness": scores["sadness"],
                    }
                }
            ]
        }

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=MockResponse(json.dumps(payload)),
        ):
            result = emotion_detector(statement)

        self.assertEqual(result["dominant_emotion"], expected_dominant)

    def test_joy(self):
        self._run_case(
            "I am glad this happened",
            "joy",
            {"anger": 0.05, "disgust": 0.03, "fear": 0.04, "joy": 0.82, "sadness": 0.06},
        )

    def test_anger(self):
        self._run_case(
            "I am really mad about this",
            "anger",
            {"anger": 0.84, "disgust": 0.05, "fear": 0.04, "joy": 0.02, "sadness": 0.05},
        )

    def test_disgust(self):
        self._run_case(
            "I feel disgusted just hearing about this",
            "disgust",
            {"anger": 0.08, "disgust": 0.81, "fear": 0.04, "joy": 0.02, "sadness": 0.05},
        )

    def test_sadness(self):
        self._run_case(
            "I am so sad about this",
            "sadness",
            {"anger": 0.06, "disgust": 0.04, "fear": 0.08, "joy": 0.02, "sadness": 0.80},
        )

    def test_fear(self):
        self._run_case(
            "I am really afraid that this will happen",
            "fear",
            {"anger": 0.05, "disgust": 0.04, "fear": 0.83, "joy": 0.03, "sadness": 0.05},
        )


if __name__ == "__main__":
    unittest.main()
