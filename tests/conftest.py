"""Tests always own their runtime/config/media data, never the user's monitor."""
import os
import shutil
import tempfile
from pathlib import Path
import pytest

_sandbox = tempfile.TemporaryDirectory(prefix='studio_tests_')
_root = Path(_sandbox.name)
os.environ['STUDIO_RUNTIME_DIR'] = str(_root / 'runtime')
os.environ['TELEGRAM_BOT_TOKEN'] = ''
import config
config.TELEGRAM_BOT_TOKEN = ''
config.UPLOAD_FOLDER = _root / 'uploads'
config.OUTPUT_FOLDER = _root / 'outputs'
config.UPLOAD_FOLDER.mkdir()
config.OUTPUT_FOLDER.mkdir()
config.PRESETS_FILE = _root / 'presets.json'
shutil.copyfile(config.BASE_DIR / 'presets.json', config.PRESETS_FILE)
import services.douyin_monitor.channel_manager as cm
cm.CONFIG_DIR = _root / 'config'
cm.CONFIG_DIR.mkdir()
cm.CHANNELS_FILE = cm.CONFIG_DIR / 'monitor_channels.json'
cm.HISTORY_FILE = cm.CONFIG_DIR / 'downloaded_history.json'
cm.NOTIFY_CHAT_FILE = cm.CONFIG_DIR / 'telegram_notify_chat.json'
cm.DEFAULT_CHANNELS = []
cm.CHANNELS_FILE.write_text('[]', encoding='utf-8')


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(_root / 'pytest')


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import routes.api as api
    # Route tests exercise dispatch contracts. Actual worker tests call the worker
    # directly, so a daemon cannot outlive a mock and contact a real provider.
    monkeypatch.setattr(api, 'threading', SimpleNamespace(Thread=MagicMock(), RLock=__import__('threading').RLock))
    config.jobs.clear()
    for jid in config.jobs.store.ids():
        config.jobs.store.delete(jid)
    yield
