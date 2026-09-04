import os

from ingestion.download import download_source


def test_download_source_skips_when_file_exists(tmp_path):
    existing = tmp_path / "already-here.html"
    existing.write_text("<html>cached</html>", encoding="utf-8")

    result = download_source(str(existing), url="https://example.invalid/should-not-be-called")

    assert result == str(existing)
    assert existing.read_text(encoding="utf-8") == "<html>cached</html>"


def test_download_source_fetches_when_missing(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "fetched.html"

    class _FakeResponse:
        content = b"<html>fetched</html>"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        assert url == "https://example.invalid/source"
        return _FakeResponse()

    monkeypatch.setattr("ingestion.download.requests.get", fake_get)

    result = download_source(str(target), url="https://example.invalid/source")

    assert result == str(target)
    assert os.path.exists(target)
    assert target.read_bytes() == b"<html>fetched</html>"
