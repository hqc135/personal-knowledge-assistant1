import sys
import logging
from pathlib import Path
from data_pipeline import DocumentProcessor, SemanticChunker

logging.basicConfig(level=logging.INFO)

test_markdown = """
这是一段测试文本，用来验证代码是否被正确处理。

```python
def hello_world():
    print("Hello, world!")
    return True
```

这是代码块后面的文本。
"""

def test_pipeline():
    print("=== Testing SemanticChunker ===")
    from embedder import ZhipuEmbedder
    embedder = ZhipuEmbedder()
    chunker = SemanticChunker(embedder)
    
    chunks = chunker.split_text(test_markdown)
    for i, c in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  is_code: {c.get('is_code')}")
        print(f"  code_language: {c.get('code_language')}")
        print(f"  text_length: {len(c['text'])}")
        print(f"  text_preview: {c['text'][:50]!r}")
        print("-" * 40)

    print("\n=== Testing DocumentProcessor._split_document ===")
    processor = DocumentProcessor()
    docs, chunker_name = processor._split_document(test_markdown, "test.md")
    for i, d in enumerate(docs):
        print(f"Doc {i} [{chunker_name}]:")
        print(f"  is_code: {d.get('is_code')}")
        print(f"  code_language: {d.get('code_language')}")
        print("-" * 40)

if __name__ == "__main__":
    test_pipeline()
