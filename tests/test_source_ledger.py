from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from services.douyin_monitor.ledger import SourceLedger

class SourceLedgerTests(unittest.TestCase):
    def test_claim_is_idempotent_and_failure_can_be_reclaimed(self):
        with TemporaryDirectory() as tmp:
            ledger=SourceLedger(Path(tmp)/"ledger.sqlite3")
            work={"aweme_id":"123","desc":"demo"}
            self.assertTrue(ledger.claim(channel_id="ch",work=work))
            self.assertFalse(ledger.claim(channel_id="ch",work=work))
            ledger.mark(channel_id="ch",work_id="123",status="FAILED",error="render")
            self.assertTrue(ledger.claim(channel_id="ch",work=work))
            self.assertEqual(ledger.get("ch","123")["status"],"CLAIMED")

if __name__ == "__main__": unittest.main()
