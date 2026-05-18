"""
单元测试：data_pipeline 核心逻辑
"""
import tempfile
from pathlib import Path

import pytest


class TestMakeChunkId:
    """测试 chunk ID 生成逻辑"""

    @staticmethod
    def _make_chunk_id(relative_path: Path, chunk_index: int) -> str:
        """复制自 DocumentProcessor._make_chunk_id（避免 import config 副作用）"""
        parts = list(relative_path.with_suffix("").parts)
        return "_".join(parts) + f"_{chunk_index}"

    def test_simple_file(self):
        """单层文件名"""
        result = self._make_chunk_id(Path("rag-notes.md"), 0)
        assert result == "rag-notes_0"

    def test_nested_path(self):
        """子目录下的文件不应和根目录同名文件碰撞"""
        id_root = self._make_chunk_id(Path("design.md"), 0)
        id_sub1 = self._make_chunk_id(Path("sub1/design.md"), 0)
        id_sub2 = self._make_chunk_id(Path("sub2/design.md"), 0)

        assert id_root == "design_0"
        assert id_sub1 == "sub1_design_0"
        assert id_sub2 == "sub2_design_0"
        # 关键：三个 ID 互不相同
        assert len({id_root, id_sub1, id_sub2}) == 3

    def test_different_chunks_same_file(self):
        """同一文件不同 chunk 的 ID 不同"""
        ids = [self._make_chunk_id(Path("notes.md"), i) for i in range(5)]
        assert len(set(ids)) == 5

    def test_deeply_nested(self):
        """深层嵌套路径"""
        result = self._make_chunk_id(Path("a/b/c/deep.md"), 3)
        assert result == "a_b_c_deep_3"


class TestLoadMarkdown:
    """测试 Markdown 文件加载和分块"""

    def test_load_single_file(self, tmp_path: Path):
        """加载单个 markdown 文件"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nSome content here.", encoding="utf-8")

        # 手动模拟 load 逻辑（不依赖完整 DocumentProcessor）
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""],
        )

        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert "Title" in chunks[0]

    def test_empty_directory(self, tmp_path: Path):
        """空目录应返回空列表"""
        files = list(tmp_path.rglob("*.md"))
        assert files == []

    def test_file_hash_deterministic(self, tmp_path: Path):
        """同一文件内容的 hash 应一致"""
        import hashlib

        f = tmp_path / "test.md"
        f.write_text("hello world", encoding="utf-8")
        h1 = hashlib.md5(f.read_bytes()).hexdigest()
        h2 = hashlib.md5(f.read_bytes()).hexdigest()
        assert h1 == h2

    def test_file_hash_changes(self, tmp_path: Path):
        """文件修改后 hash 应变化"""
        import hashlib

        f = tmp_path / "test.md"
        f.write_text("version 1", encoding="utf-8")
        h1 = hashlib.md5(f.read_bytes()).hexdigest()
        f.write_text("version 2", encoding="utf-8")
        h2 = hashlib.md5(f.read_bytes()).hexdigest()
        assert h1 != h2
