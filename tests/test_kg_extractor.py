"""
单元测试：kg_extractor 解析逻辑
"""

import os

os.environ.setdefault("ZHIPUAI_API_KEY", "test-zhipu-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")

import kg_extractor
from kg_extractor import _extract_json, _normalize_item


class TestExtractJson:
    def test_plain_json_object(self):
        payload = '{"triples": [{"head": "A", "relation": "rel", "tail": "B"}]}'
        result = _extract_json(payload)
        assert result["triples"][0]["head"] == "A"

    def test_fenced_json(self):
        payload = """```json
        [{"head": "A", "relation": "rel", "tail": "B"}]
        ```"""
        result = _extract_json(payload)
        assert result[0]["tail"] == "B"

    def test_json_with_leading_text(self):
        payload = "Here is the result: [{\"head\": \"A\", \"relation\": \"rel\", \"tail\": \"B\"}]"
        result = _extract_json(payload)
        assert result[0]["relation"] == "rel"


class TestNormalizeItem:
    def test_accepts_alias_keys(self):
        normalized = _normalize_item({"entity1": "A", "predicate": "rel", "entity2": "B"})
        assert normalized == {"head": "A", "relation": "rel", "tail": "B"}

    def test_rejects_incomplete_item(self):
        assert _normalize_item({"head": "A", "relation": "rel"}) is None


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class _FakeCompletions:
    def __init__(self, content: str | None = None, messages: list[object] | None = None):
        self._content = content
        self._messages = messages or []
        self.last_kwargs = None
        self.calls = 0

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls += 1
        if self._messages:
            return self._messages.pop(0)
        return _FakeResponse(self._content or "")


class _FakeClient:
    def __init__(self, content: str | None = None, messages: list[object] | None = None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, messages)})()


def _make_response(content: str, *, finish_reason: str = "stop", reasoning_content: str | None = None):
    message = type(
        "Message",
        (),
        {
            "content": content,
            "reasoning_content": reasoning_content,
        },
    )()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    return type("Response", (), {"choices": [choice]})()


class TestExtractTriples:
    def test_extract_triples_with_structured_json(self, monkeypatch):
        fake_client = _FakeClient(
            '{"triples": [{"head": "A", "relation": "rel", "tail": "B"}]}'
        )
        monkeypatch.setattr(kg_extractor, "_create_client", lambda: fake_client)

        result = kg_extractor.extract_triples("text", source="notes/a.md", chunk_id="chunk-1")

        assert result == [
            {
                "head": "A",
                "relation": "rel",
                "tail": "B",
                "source": "notes/a.md",
                "chunk_id": "chunk-1",
            }
        ]

    def test_extract_triples_rejects_non_list_payload(self, monkeypatch):
        fake_client = _FakeClient('{"triples": {"head": "A"}}')
        monkeypatch.setattr(kg_extractor, "_create_client", lambda: fake_client)

        assert kg_extractor.extract_triples("text", source="notes/a.md", chunk_id="chunk-1") == []

    def test_fallback_to_non_reasoning_model(self, monkeypatch):
        responses = [
            _make_response("", finish_reason="length", reasoning_content="thinking"),
            _make_response('{"triples": [{"head": "A", "relation": "rel", "tail": "B"}]}'),
        ]
        fake_client = _FakeClient(messages=responses)

        monkeypatch.setattr(kg_extractor, "_create_client", lambda: fake_client)
        monkeypatch.setattr(kg_extractor.config, "KG_LLM_MODEL", "deepseek-v4-flash")
        monkeypatch.setattr(kg_extractor.config, "KG_FALLBACK_MODEL", "deepseek-chat")

        result = kg_extractor.extract_triples("text", source="notes/a.md", chunk_id="chunk-1")

        assert result
        assert fake_client.chat.completions.calls == 2

    def test_postprocess_filters_short_and_duplicate(self):
        raw = [
            {"head": "A", "relation": "rel", "tail": "B", "source": "s", "chunk_id": "c1"},
            {"head": "A", "relation": "rel", "tail": "B", "source": "s", "chunk_id": "c1"},
            {"head": "X", "relation": "r", "tail": "y", "source": "s", "chunk_id": "c2"},
            {"head": "GoodEntity", "relation": " long relation    with   spaces ", "tail": "Some tail text", "source": "s", "chunk_id": "c3"},
            {"head": "S", "relation": "r", "tail": "TooShort", "source": "s", "chunk_id": "c4"},
        ]

        processed = kg_extractor._postprocess_triples(raw)
        # duplicate removed, short head 'X'/'y' removed (tail length<2 or head length<2)
        # expect entries: GoodEntity and maybe S filtered by head length
        assert any(t["head"] == "GoodEntity" for t in processed)
        # duplicates reduced
        heads = [t["head"] for t in processed]
        assert heads.count("A") == 1 or "A" not in heads