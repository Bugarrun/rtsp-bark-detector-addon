import time
import logging


class BarkDetector:

    def __init__(self, bark_threshold, dog_threshold, release_seconds):
        self.bark_threshold = bark_threshold
        self.dog_threshold = dog_threshold
        self.release_seconds = release_seconds

        self.barking_state = False
        self.last_bark_time = None
        self.episode_start_time = None
        self.confidences = []

        self.logger = logging.getLogger("dog_bark_detector")


    def update(self, bark_score, dog_score):

        event = None

        now = time.time()

        is_barking = (
            bark_score >= self.bark_threshold
            and dog_score >= self.dog_threshold
        )

        if is_barking:
            self.last_bark_time = now

            if not self.barking_state:
                self.barking_state = True
                self.episode_start_time = now
                self.confidences = []
                
                event = {
                    "event": "started",
                    "state": True,
                    "timestamp": now
                }

                self.logger.info("DOG BARKING STARTED")

            self.confidences.append(
                (bark_score + dog_score) / 2
            )

        elif self.barking_state and self.last_bark_time:
            if now - self.last_bark_time > self.release_seconds:

                duration = now - self.episode_start_time

                if self.confidences:
                    average_confidence = sum(self.confidences) / len(self.confidences)
                else:
                    average_confidence = 0

                self.barking_state = False

                event = {
                    "event": "stopped",
                    "state": False,
                    "duration": round(duration, 2),
                    "confidence": round(average_confidence, 2),
                    "timestamp": now
                }

                self.logger.info(
                    f"DOG BARKING STOPPED | "
                    f"Duration={duration:.1f}s "
                    f"Average confidence={average_confidence:.2f}"
                )

                self.episode_start_time = None
                self.confidences = []

        return event