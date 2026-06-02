"""Unit tests for generator prompt formatting."""

from generator import Generator


class TestGeneratorPrompt:
    def test_build_messages_includes_evidence_chain(self):
        generator = Generator.__new__(Generator)
        contexts = [
            {
                "text": "chunk body",
                "metadata": {"source": "notes/demo.md", "chunk_index": 3, "vector_tag": "v"},
                "score": 0.8123,
                "channels": ["bm25", "vector"],
                "channel_scores": {"vector": 0.91234, "bm25": 2.5},
            }
        ]

        messages = Generator._build_messages(generator, "what is this", contexts)

        assert messages[0]["role"] == "system"
        assert "raw_scores" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "上下文证据链:" in messages[1]["content"]
        assert "[来源: notes/demo.md]" in messages[1]["content"]
        assert "[通道: bm25, vector]" in messages[1]["content"]
        assert "raw_scores: bm25=2.5000, vector=0.9123" in messages[1]["content"]
        assert "[fused_score: 0.8123]" in messages[1]["content"]
        assert "vector_tag" in messages[1]["content"]
