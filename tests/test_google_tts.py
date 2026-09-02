import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services import google_tts


class GoogleTTSTests(unittest.TestCase):
    def make_settings(self, keys=("test-key",)):
        return google_tts.GoogleTTSSettings(
            endpoint="https://generativelanguage.googleapis.com/v1beta/interactions",
            model="gemini-3.1-flash-tts-preview",
            voice="Charon",
            sample_rate=24000,
            timeout_seconds=10,
            default_emotion="warm",
            style_prompt="Cinematic narration.",
            language_profiles=dict(google_tts.DEFAULT_LANGUAGE_PROFILES),
            keys=keys,
            credential_source="test",
        )

    def test_payload_uses_model_voice_locale_and_emotion(self):
        payload = google_tts.build_interaction_payload(
            "Đây là lời dẫn.",
            lang="vi",
            voice="Kore",
            emotion="dramatic",
            settings=self.make_settings(),
        )
        self.assertEqual(payload["model"], "gemini-3.1-flash-tts-preview")
        self.assertEqual(payload["generation_config"]["speech_config"], [{"voice": "Kore"}])
        self.assertIn("vi-VN", payload["input"])
        self.assertIn("dramatic", payload["input"].lower())

    def test_markup_is_sanitized_and_cues_become_directions(self):
        clean, cues = google_tts.compile_performance(
            "[mystery] Xin chào [laughs] [unknown]."
        )
        self.assertNotIn("[mystery]", clean)
        self.assertNotIn("[unknown]", clean)
        self.assertIn("[laughing]", clean)
        self.assertEqual(cues, ["mystery"])

    def test_credential_precedence_and_health_do_not_expose_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            media_path = Path(temp_dir) / "google_media.json"
            media_path.write_text(json.dumps({"api_key": "file-secret"}), encoding="utf-8")
            with patch.object(google_tts, "GOOGLE_MEDIA_CONFIG", media_path), \
                 patch.dict("os.environ", {"GEMINI_API_KEY": "env-secret"}, clear=False):
                keys, source = google_tts.load_google_media_credentials()
                self.assertEqual(keys, ("env-secret",))
                self.assertEqual(source, "GEMINI_API_KEY")
                with patch.object(google_tts, "load_google_tts_settings", return_value=self.make_settings(keys)):
                    health = google_tts.get_google_tts_health()
                self.assertNotIn("env-secret", json.dumps(health))

    def test_synthesize_writes_pcm_wav(self):
        pcm = b"\x00\x00" * 1200
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "steps": [{"type": "model_output", "content": [{
                "type": "audio", "data": base64.b64encode(pcm).decode("ascii")
            }]}]
        }
        client = MagicMock()
        client.__enter__.return_value.post.return_value = response
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(google_tts, "load_google_tts_settings", return_value=self.make_settings()), \
             patch.object(google_tts.httpx, "Client", return_value=client):
            output = Path(temp_dir) / "sample.wav"
            google_tts.synthesize_to_wav("Hello", output, lang="en")
            self.assertTrue(output.read_bytes().startswith(b"RIFF"))
            call = client.__enter__.return_value.post.call_args
            self.assertEqual(call.kwargs["headers"]["x-goog-api-key"], "test-key")


if __name__ == "__main__":
    unittest.main()
