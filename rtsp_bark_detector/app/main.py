import json
import logging
import math
import os
import shutil
import signal
import time

import yaml
import numpy as np

from audio_stream import AudioStream, AudioStreamError
from classifier import BarkClassifier
from detector import BarkDetector
from ha_bridge import HABridge

logger = logging.getLogger("bark_addon")


def load_config():
    with open("config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if os.path.exists("/data/options.json"):
        with open("/data/options.json", encoding="utf-8") as file:
            options = json.load(file)
        for option, section, key in (
            ("rtsp_url", "camera", "rtsp_url"),
            ("bark_threshold", "thresholds", "bark"),
            ("dog_threshold", "thresholds", "dog"),
            ("bark_release_seconds", "settings", "bark_release_seconds"),
        ):
            config[section][key] = options.get(option, config[section][key])
    config["mqtt"] = {
        "broker": os.environ["MQTT_HOST"],
        "port": int(os.environ["MQTT_PORT"]),
        "username": os.environ["MQTT_USERNAME"],
        "password": os.environ["MQTT_PASSWORD"],
    }
    if not config["camera"]["rtsp_url"].startswith(("rtsp://", "rtsps://")):
        raise ValueError("Set a valid rtsp_url in the add-on configuration")
    for key in ("bark", "dog"):
        value = float(config["thresholds"][key])
        if not 0 <= value <= 1:
            raise ValueError("Bark and dog thresholds must be between 0 and 1")
        config["thresholds"][key] = value
    release = float(config["settings"]["bark_release_seconds"])
    if not math.isfinite(release) or release < 0:
        raise ValueError("bark_release_seconds must be non-negative")
    config["settings"]["bark_release_seconds"] = release
    return config


def shutdown_handler(signum, frame):
    logger.info("Stopping RTSP Bark Detector...")
    raise SystemExit(0)


def audio_windows(stream, window_samples, hop_samples=7680):
    """Consume every PCM sample, analyzing 975 ms with a 480 ms stride."""
    if not 0 < hop_samples <= window_samples:
        raise ValueError("Audio hop must fit inside the analysis window")
    pcm = stream.read(window_samples * 2)
    yield pcm
    while True:
        pcm = pcm[hop_samples * 2:] + stream.read(hop_samples * 2)
        yield pcm


def run():
    config = load_config()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    classifier = BarkClassifier(config["settings"]["model"])
    logger.info("YAMNet ready | %s samples at %s Hz | Bark=%s Dog=%s",
                classifier.samples_per_window, classifier.sample_rate,
                classifier.bark_index, classifier.dog_index)
    detector = BarkDetector(config["thresholds"]["bark"], config["thresholds"]["dog"],
                            config["settings"]["bark_release_seconds"])
    ha = HABridge(config["mqtt"], config["device"])
    stream = None
    try:
        logger.info("RTSP Bark Detector ready")
        while True:
            try:
                logger.info("Connecting to RTSP audio...")
                stream = AudioStream(ffmpeg, config["camera"]["rtsp_url"])
                first_chunk = True
                next_score_log = 0
                peak_bark = peak_dog = peak_audio = max_inference_ms = 0.0
                windows = 0
                logger.info("Continuous audio | window=975 ms hop=480 ms | thresholds Bark=%.2f Dog=%.2f",
                            detector.bark_threshold, detector.dog_threshold)
                for pcm in audio_windows(stream, classifier.samples_per_window):
                    if first_chunk:
                        logger.info("RTSP audio stream active")
                        first_chunk = False
                    inference_start = time.monotonic()
                    bark_score, dog_score = classifier.classify_pcm(pcm)
                    max_inference_ms = max(max_inference_ms, (time.monotonic() - inference_start) * 1000)
                    windows += 1
                    if bark_score >= peak_bark:
                        peak_bark, peak_dog = bark_score, dog_score
                    waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    peak_audio = max(peak_audio, float(np.sqrt(np.mean(waveform * waveform))))
                    if bark_score >= 0.10:
                        logger.info("Bark candidate | Bark=%.4f Dog=%.4f | accepted=%s",
                                    bark_score, dog_score,
                                    bark_score >= detector.bark_threshold and dog_score >= detector.dog_threshold)
                    now = time.monotonic()
                    if now >= next_score_log:
                        logger.info("Audio active | windows=%d | peak Bark=%.4f Dog=%.4f | audio=%.1f dBFS | inference max=%.1f ms",
                                    windows, peak_bark, peak_dog,
                                    20 * math.log10(max(peak_audio, 1e-10)), max_inference_ms)
                        if max_inference_ms >= 480:
                            logger.warning("Classifier slower than the audio stride; detection may lag behind live audio")
                        next_score_log = now + 5
                        peak_bark = peak_dog = peak_audio = max_inference_ms = 0.0
                        windows = 0
                    event = detector.update(bark_score, dog_score)
                    if event:
                        ha.process_event(event)
            except AudioStreamError as error:
                logger.warning("%s; reconnecting in 5 seconds", error)
            finally:
                if stream is not None:
                    stream.close()
                    stream = None
            time.sleep(5)
    finally:
        ha.client.disconnect()
        ha.client.loop_stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    try:
        run()
    except Exception:
        logger.exception("RTSP Bark Detector failed")
        raise SystemExit(1)
