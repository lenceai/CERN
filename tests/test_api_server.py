from fastapi.testclient import TestClient

from cern_mag_llmops.api import server


class DummyRAGModel:
    def __init__(self):
        self.last_k = None

    def answer_question(self, question):
        return {
            "answer": f"RAG answer for: {question}",
            "model": "dummy-rag",
            "response_time": 0.1,
            "sources": [
                {
                    "filename": "source.pdf",
                    "year": "2024",
                    "issue": "1",
                    "excerpt": "excerpt",
                }
            ],
        }

    def get_relevant_documents(self, query, k=None):
        self.last_k = k
        return [
            {
                "content": f"doc for {query}",
                "metadata": {
                    "filename": "source.pdf",
                    "year": "2024",
                    "issue": "1",
                    "volume": "2",
                    "chunk_len": 42,
                },
            }
        ]


class DummyComparison:
    def query_fine_tuned_model(self, question):
        return {"answer": f"FT answer for: {question}", "response_time": 0.2, "model": "dummy-ft"}

    def compare_models(self, question):
        return {
            "rag": {
                "answer": f"RAG compare answer for: {question}",
                "model": "dummy-rag",
                "response_time": 0.1,
                "sources": [],
            },
            "fine_tuned": {
                "answer": f"FT compare answer for: {question}",
                "model": "dummy-ft",
                "response_time": 0.2,
            },
        }


def _make_test_client(dummy_rag):
    server.app.dependency_overrides[server.get_rag_model] = lambda: dummy_rag
    return TestClient(server.app)


def test_documents_endpoint_passes_requested_k(monkeypatch):
    dummy_rag = DummyRAGModel()
    monkeypatch.setattr(server.settings, "API_AUTH_ENABLED", False)
    client = _make_test_client(dummy_rag)

    response = client.post("/documents", json={"query": "higgs", "k": 3})

    assert response.status_code == 200
    assert dummy_rag.last_k == 3

    server.app.dependency_overrides.clear()


def test_query_rag_does_not_initialize_model_comparison(monkeypatch):
    dummy_rag = DummyRAGModel()
    monkeypatch.setattr(server.settings, "API_AUTH_ENABLED", False)
    def _unexpected_model_comparison():
        raise RuntimeError("model comparison should not be initialized for rag queries")

    monkeypatch.setattr(server, "get_model_comparison", _unexpected_model_comparison)
    client = _make_test_client(dummy_rag)

    response = client.post("/query", json={"question": "What is CERN?", "model_type": "rag"})

    assert response.status_code == 200
    assert response.json()["rag"]["answer"].startswith("RAG answer")

    server.app.dependency_overrides.clear()


def test_api_key_enforced_when_enabled(monkeypatch):
    dummy_rag = DummyRAGModel()
    monkeypatch.setattr(server.settings, "API_AUTH_ENABLED", True)
    monkeypatch.setattr(server.settings, "API_AUTH_KEY", "secret")
    client = _make_test_client(dummy_rag)

    unauthorized = client.post("/documents", json={"query": "higgs", "k": 1})
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/documents",
        json={"query": "higgs", "k": 1},
        headers={"X-API-Key": "secret"},
    )
    assert authorized.status_code == 200

    server.app.dependency_overrides.clear()


def test_query_fine_tuned_uses_model_comparison(monkeypatch):
    dummy_rag = DummyRAGModel()
    monkeypatch.setattr(server.settings, "API_AUTH_ENABLED", False)
    monkeypatch.setattr(server.settings, "FINE_TUNED_MODEL", "ft-model")
    monkeypatch.setattr(server, "get_model_comparison", lambda: DummyComparison())
    client = _make_test_client(dummy_rag)

    response = client.post("/query", json={"question": "What is CERN?", "model_type": "fine_tuned"})

    assert response.status_code == 200
    assert response.json()["fine_tuned"]["model"] == "dummy-ft"

    server.app.dependency_overrides.clear()
