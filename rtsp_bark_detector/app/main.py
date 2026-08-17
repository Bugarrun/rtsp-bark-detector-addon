import subprocess
import logging
import time
import yaml
import numpy as np
import shutil
import signal

from tflite_runtime.interpreter import Interpreter

from detector import BarkDetector
from ha_bridge import HABridge


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

logger = logging.getLogger("bark_addon")


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


RTSP_URL = config["camera"]["rtsp_url"]

BARK_THRESHOLD = config["thresholds"]["bark"]
DOG_THRESHOLD = config["thresholds"]["dog"]
BARK_RELEASE_SECONDS = config["settings"]["bark_release_seconds"]


MODEL = config["settings"]["model"]

FFMPEG = shutil.which("ffmpeg")

if not FFMPEG:
    FFMPEG = ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"

if not FFMPEG:
    raise RuntimeError("FFmpeg not found")


detector = BarkDetector(
    BARK_THRESHOLD,
    DOG_THRESHOLD,
    BARK_RELEASE_SECONDS
)

ha = HABridge(
    config["mqtt"],
    config["device"]
)



interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logger.info("RTSP Bark Detector ready")

cmd = [
    FFMPEG,
    "-rtsp_transport", "tcp",
    "-i", RTSP_URL,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-f", "s16le",
    "-"
]


logger.info("Connecting to RTSP audio...")


process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)


logger.info("Listening for barking...")


def shutdown_handler(signum, frame):

    logger.info("Stopping RTSP Bark Detector...")

    if process:
        process.terminate()
        process.wait()

    logger.info("Shutdown complete")

    raise SystemExit


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


while True:

    audio_chunk = process.stdout.read(16000 * 2)

    if not audio_chunk:
        time.sleep(0.1)
        continue


    audio = np.frombuffer(
        audio_chunk,
        dtype=np.int16
    )

    audio = audio.astype(np.float32) / 32768.0


    audio_data = AudioData.create_from_array(
        audio,
        sample_rate=16000
    )


    result = classifier.classify(audio_data)


    bark_score = 0
    dog_score = 0


    for category in result[0].classifications[0].categories:

        if category.category_name == "Bark":
            bark_score = category.score

        elif category.category_name == "Dog":
            dog_score = category.score


    event = detector.update(
        bark_score,
        dog_score
    )


    if event:
        ha.process_event(event)
