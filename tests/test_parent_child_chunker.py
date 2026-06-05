"""
parent_child_chunker.py 的单元测试。

覆盖场景：
  - Markdown 标题切分 parent
  - 无标题文档的 fallback
  - Child 与 Parent 的 ID 关联正确性
  - 空文档 / 极短文档
  - 过长 section 的拆分
  - 过短 section 的合并
"""
import unittest
from unittest.mock import patch, MagicMock


class TestSplitByHeadings(unittest.TestCase):
    """测试 _split_by_headings 内部函数。"""

    def test_single_heading(self):
        from parent_child_chunker import _split_by_headings

        text = "## 标题一\n\n这是第一段内容。\n\n一些文本。"
        sections = _split_by_headings(text)
        self.assertEqual(len(sections), 1)
        self.assertIn("标题一", sections[0])

    def test_multiple_headings(self):
        from parent_child_chunker import _split_by_headings

        text = "# 大标题\n\n介绍内容。\n\n## 标题一\n\n第一段。\n\n## 标题二\n\n第二段。"
        sections = _split_by_headings(text)
        self.assertEqual(len(sections), 3)
        self.assertIn("大标题", sections[0])
        self.assertIn("标题一", sections[1])
        self.assertIn("标题二", sections[2])

    def test_no_headings(self):
        from parent_child_chunker import _split_by_headings

        text = "这是一段没有标题的纯文本。\n\n第二段。"
        sections = _split_by_headings(text)
        self.assertEqual(len(sections), 1)
        self.assertIn("纯文本", sections[0])

    def test_empty_text(self):
        from parent_child_chunker import _split_by_headings

        sections = _split_by_headings("")
        self.assertEqual(len(sections), 0)

    def test_preamble_before_heading(self):
        from parent_child_chunker import _split_by_headings

        text = "前言内容\n\n## 标题\n\n正文"
        sections = _split_by_headings(text)
        self.assertEqual(len(sections), 2)
        self.assertIn("前言", sections[0])
        self.assertIn("标题", sections[1])


class TestMergeSmallSections(unittest.TestCase):
    """测试 _merge_small_sections 内部函数。"""

    def test_merge_small_forward(self):
        from parent_child_chunker import _merge_small_sections

        sections = ["短", "也很短", "这是一段足够长的内容" * 20]
        result = _merge_small_sections(sections, min_size=50, max_size=5000)
        # 前两个短 section 应被合并
        self.assertTrue(len(result) <= 2)

    def test_split_oversized(self):
        from parent_child_chunker import _merge_small_sections

        long_section = "这是一段非常长的内容。" * 500  # 约 5000 chars
        result = _merge_small_sections([long_section], min_size=50, max_size=500)
        self.assertTrue(len(result) > 1)
        for chunk in result:
            # 允许一点超出（splitter 有 overlap 等），但大致应在 max_size 附近
            self.assertLessEqual(len(chunk), 800)


class TestParentChildChunker(unittest.TestCase):
    """测试 ParentChildChunker 完整流程。"""

    @patch("parent_child_chunker.config")
    def _make_chunker(self, mock_config):
        """构造一个不依赖真实 embedder 的 chunker（使用 fallback splitter）。"""
        mock_config.PARENT_CHUNK_MIN_SIZE = 100
        mock_config.PARENT_CHUNK_MAX_SIZE = 2000
        mock_config.CHILD_CHUNK_MIN_SIZE = 30
        mock_config.CHILD_CHUNK_MAX_SIZE = 200
        mock_config.SEMANTIC_CHUNKER_ENABLED = False  # 强制使用 fallback
        mock_config.CHUNK_SIZE = 200
        mock_config.CHUNK_OVERLAP = 20

        from parent_child_chunker import ParentChildChunker
        return ParentChildChunker(embedder=None)

    def test_basic_split(self):
        """基本 Parent-Child 切分：有标题的 Markdown 文档。"""
        chunker = self._make_chunker()

        text = (
            "# 主标题\n\n这是一段介绍。这段介绍足够长，可以形成一个 parent chunk。"
            "额外的内容让它超过 min_size。" * 5
            + "\n\n## 子标题一\n\n第一节的内容。" * 10
            + "\n\n## 子标题二\n\n第二节的内容。" * 10
        )

        parents, children = chunker.split_document(text, "test.md")

        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), 0)

        # 验证 parent ID 格式
        for p in parents:
            self.assertIn("_parent_", p["id"])
            self.assertEqual(p["metadata"]["chunk_type"], "parent")
            self.assertEqual(p["metadata"]["source"], "test.md")

        # 验证 child 的 parent_id 关联
        parent_ids = {p["id"] for p in parents}
        for c in children:
            self.assertIn("_child_", c["id"])
            self.assertEqual(c["metadata"]["chunk_type"], "child")
            self.assertIn(c["metadata"]["parent_id"], parent_ids)

    def test_no_heading_document(self):
        """无标题文档应仍能产生 parent 和 child。"""
        chunker = self._make_chunker()

        text = "这是一段没有标题的长文本。\n\n" * 20

        parents, children = chunker.split_document(text, "plain.md")

        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), 0)

    def test_empty_document(self):
        """空文档应返回空列表。"""
        chunker = self._make_chunker()

        parents, children = chunker.split_document("", "empty.md")
        self.assertEqual(len(parents), 0)
        self.assertEqual(len(children), 0)

    def test_short_document(self):
        """极短文档应至少产生一个 parent。"""
        chunker = self._make_chunker()

        text = "# 标题\n\n简短内容。"
        parents, children = chunker.split_document(text, "short.md")

        self.assertGreater(len(parents), 0)

    def test_child_count_metadata(self):
        """parent 的 child_count 应与实际 child 数匹配。"""
        chunker = self._make_chunker()

        text = "## 一个标题\n\n" + "这是足够长的内容用于分块。" * 30

        parents, children = chunker.split_document(text, "count.md")

        for p in parents:
            expected_count = p["metadata"]["child_count"]
            actual = sum(
                1 for c in children if c["metadata"]["parent_id"] == p["id"]
            )
            self.assertEqual(expected_count, actual)


class TestParentChildIds(unittest.TestCase):
    """测试 ID 生成函数。"""

    def test_make_parent_id(self):
        from pathlib import Path
        from parent_child_chunker import _make_parent_id

        pid = _make_parent_id(Path("chunking-strategies.md"), 0)
        self.assertEqual(pid, "chunking-strategies_parent_0")

    def test_make_child_id(self):
        from pathlib import Path
        from parent_child_chunker import _make_child_id

        cid = _make_child_id(Path("sub/design.md"), 1, 3)
        self.assertEqual(cid, "sub_design_child_1_3")

    def test_make_parent_id_nested_path(self):
        from pathlib import Path
        from parent_child_chunker import _make_parent_id

        pid = _make_parent_id(Path("notes/deep/file.md"), 2)
        self.assertEqual(pid, "notes_deep_file_parent_2")


if __name__ == "__main__":
    unittest.main()
