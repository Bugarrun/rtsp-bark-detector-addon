import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rtsp_bark_detector/app'))
from audio_stream import AudioStream, AudioStreamError, redact
from classifier import BarkClassifier


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = BarkClassifier(ROOT / 'rtsp_bark_detector/models/yamnet.tflite')

    def test_silence_and_partial_window(self):
        self.assertEqual(self.classifier.classify_pcm(bytes(31200)), (0.0, 0.0))
        for pcm in (b'', b'\0', bytes(32000)):
            with self.assertRaises(ValueError):
                self.classifier.classify_pcm(pcm)

    def stream_for(self, program):
        proc = subprocess.Popen([sys.executable, '-c', program], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with patch('audio_stream.subprocess.Popen', return_value=proc):
            stream = AudioStream('ffmpeg', 'rtsp://user:secret@camera/audio')
        self.addCleanup(stream.close)
        return stream

    def test_partial_reads_and_eof(self):
        stream = self.stream_for("import os,time; os.write(1,b'ab'); time.sleep(.05); os.write(1,b'cdef')")
        self.assertEqual(stream.read(6), b'abcdef')
        with self.assertRaises(AudioStreamError):
            stream.read(2)

    def test_stalled_stream_times_out(self):
        stream = self.stream_for('import time; time.sleep(10)')
        with self.assertRaises(AudioStreamError):
            stream.read(2, timeout=.05)
        stream.close()
        self.assertIsNotNone(stream.process.poll())
        self._cleanups.clear()

    def test_credentials_redacted(self):
        url = 'rtsp://user:secret@camera/audio'
        self.assertEqual(redact('failed '+url, url), 'failed [RTSP URL]')
        self.assertNotIn('secret', redact('failed rtsp://user:secret@camera/other', url))


if __name__ == '__main__':
    unittest.main()
