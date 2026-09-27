from __future__ import annotations
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.publish_outbox import emit_publish_package

class PublishOutboxTests(unittest.TestCase):
    def test_emit_ready_manifest_is_atomic_and_import_compatible(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            video = root / "localized.mp4"
            video.write_bytes(b"localized")
            outbox = root / "outbox"
            result = {
                "video_path": str(video),
                "translation_method": "ai",
                "translation_fallbacks": [],
                "translation_provider_warning": None,
            }
            item = {
                "item_id": "item-1",
                "source_path": str(source),
                "relative_path": "source.mp4",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            path = emit_publish_package(
                batch_id="batch-1", item=item, result=result,
                options={"target_lang": "vi", "strict_translation": True,
                         "publish_rights_status": "licensed", "publish_niche": "beauty"},
                outbox_dir=outbox,
            )
            self.assertTrue(path.name.endswith(".ready.json"))
            self.assertFalse(any(outbox.glob("*.tmp")))
            package = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(package["rights_status"], "licensed")
            self.assertEqual(package["qc_status"], "passed")
            self.assertEqual(package["final_video"]["sha256"], hashlib.sha256(video.read_bytes()).hexdigest())
            self.assertEqual(package["translation"]["review_status"], "approved")

    def test_emit_marks_unapproved_rights_or_translation_fallback_for_review(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            video = root / "localized.mp4"
            video.write_bytes(b"localized")
            item = {"item_id": "i", "source_path": str(source), "relative_path": "source.mp4"}
            rights_path = emit_publish_package(
                batch_id="b", item=item, result={"video_path": str(video)},
                options={"strict_translation": True, "publish_rights_status": "unknown"},
                outbox_dir=root / "rights-outbox",
            )
            rights_package = json.loads(rights_path.read_text(encoding="utf-8"))
            self.assertEqual(rights_package["qc_status"], "review_required")
            self.assertEqual(rights_package["translation"]["review_status"], "approved")

            fallback_path = emit_publish_package(
                batch_id="b", item=item,
                result={"video_path": str(video), "translation_fallbacks": ["line-1"]},
                options={"strict_translation": True, "publish_rights_status": "licensed"},
                outbox_dir=root / "fallback-outbox",
            )
            fallback_package = json.loads(fallback_path.read_text(encoding="utf-8"))
            self.assertEqual(fallback_package["qc_status"], "review_required")
            self.assertEqual(fallback_package["translation"]["review_status"], "awaiting_review")


if __name__ == "__main__":
    unittest.main()
