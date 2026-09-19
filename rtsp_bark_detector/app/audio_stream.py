"""Bounded FFmpeg reads, diagnostic logging, and deterministic cleanup."""
import logging
import os
import re
import selectors
import subprocess
import threading
import time

logger = logging.getLogger("bark_addon")


class AudioStreamError(RuntimeError):
    pass


def redact(message, rtsp_url):
    message = message.replace(rtsp_url, "[RTSP URL]") if rtsp_url else message
    return re.sub(r"rtsps?://\S+", "[RTSP URL]", message, flags=re.IGNORECASE)


class AudioStream:
    def __init__(self, ffmpeg, rtsp_url):
        self.rtsp_url = rtsp_url
        self.process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
             "-rtsp_transport", "tcp",
             "-i", rtsp_url, "-map", "0:a:0", "-vn", "-ac", "1",
             "-ar", "16000", "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.log_thread = threading.Thread(target=self._log_errors, daemon=True)
        self.log_thread.start()

    def _log_errors(self):
        for line in iter(self.process.stderr.readline, b""):
            message = redact(line.decode("utf-8", errors="replace").strip(), self.rtsp_url)
            if message:
                logger.warning("FFmpeg: %s", message)

    def read(self, size, timeout=20):
        data = bytearray()
        deadline = time.monotonic() + timeout
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise AudioStreamError("Timed out waiting for a complete audio window")
            chunk = os.read(self.process.stdout.fileno(), size - len(data))
            if not chunk:
                raise AudioStreamError("FFmpeg audio stream ended")
            data.extend(chunk)
        return bytes(data)

    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log_thread.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()
