from cern_mag_llmops.data_processing.pdf_processor import PDFProcessor


def test_preprocess_text_removes_noise(tmp_path):
    processor = PDFProcessor(pdf_dir=str(tmp_path), output_dir=str(tmp_path / "out"))

    raw_text = "Page 1 of 12 CERN Courier July 2024 Particle physics! <bad> keep_this."
    cleaned = processor.preprocess_text(raw_text)

    assert "Page 1 of 12" not in cleaned
    assert "CERN Courier July 2024" not in cleaned
    assert "<" not in cleaned
    assert "Particle physics!" in cleaned


def test_chunk_text_returns_chunks_with_metadata(tmp_path):
    processor = PDFProcessor(pdf_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    metadata = {"filename": "sample.pdf", "year": "2024"}
    text = "Sentence one. Sentence two is longer. Sentence three. Sentence four."

    chunks = processor.chunk_text(text=text, metadata=metadata, chunk_size=35, overlap=10)

    assert len(chunks) >= 2
    assert all("text" in chunk for chunk in chunks)
    assert all(chunk["filename"] == "sample.pdf" for chunk in chunks)
