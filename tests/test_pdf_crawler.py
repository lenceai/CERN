from cern_mag_llmops.data_ingestion.pdf_crawler import CERNPDFCrawler
from cern_mag_llmops.config import settings


class StubCrawler(CERNPDFCrawler):
    def __init__(self, download_folder: str, outcomes: dict[str, str]) -> None:
        super().__init__(download_folder=download_folder)
        self._outcomes = outcomes

    def find_courier_links(self, page_url):  # noqa: D401
        return ["https://example.org/article-1"]

    def find_pdf_links(self, article_url):  # noqa: D401
        return list(self._outcomes.keys())

    def sanitize_filename(self, url):  # noqa: D401
        return url.rsplit("/", maxsplit=1)[-1]

    def download_pdf(self, pdf_url, filename):  # noqa: D401
        return self._outcomes[pdf_url]


def test_crawl_and_download_summary_counts_downloaded_skipped_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CRAWL_DELAY", 0.0)

    outcomes = {
        "https://example.org/first.pdf": "downloaded",
        "https://example.org/second.pdf": "skipped",
        "https://example.org/third.pdf": "failed",
    }
    crawler = StubCrawler(download_folder=str(tmp_path), outcomes=outcomes)

    summary = crawler.crawl_and_download(start_page=0, end_page=0)

    assert summary["total_found"] == 3
    assert summary["downloaded"] == 1
    assert summary["skipped"] == 1
    assert summary["failed"] == 1
    assert "third.pdf" in summary["failed_files"]
