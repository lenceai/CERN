import pytest

from cern_mag_llmops.config import settings


def test_require_openai_api_key_raises_when_missing(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        settings.require_openai_api_key()
