from types import SimpleNamespace

from cern_mag_llmops.config import settings
from cern_mag_llmops.model.rag_model import RAGModel


class DummyVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append((query, k))
        return [SimpleNamespace(page_content="document text", metadata={"filename": "doc.pdf"})]


def test_get_relevant_documents_honors_requested_k():
    model = RAGModel.__new__(RAGModel)
    model.vectorstore = DummyVectorStore()

    docs = RAGModel.get_relevant_documents(model, query="higgs", k=7)

    assert model.vectorstore.calls == [("higgs", 7)]
    assert docs[0]["content"] == "document text"


def test_get_relevant_documents_uses_default_k(monkeypatch):
    model = RAGModel.__new__(RAGModel)
    model.vectorstore = DummyVectorStore()
    monkeypatch.setattr(settings, "TOP_K_DOCUMENTS", 5)

    RAGModel.get_relevant_documents(model, query="lhc")

    assert model.vectorstore.calls == [("lhc", 5)]
