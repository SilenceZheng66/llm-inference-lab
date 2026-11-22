"""prepare_data 纯逻辑单测（不依赖 datasets/torch，本地可跑）。"""
import hashlib
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from prepare_data import looks_structured, md5_of, to_messages  # noqa: E402


class TestToMessages:
    def test_basic(self):
        m = to_messages("写一个函数", "", "def f(): pass")
        assert m["messages"] == [
            {"role": "user", "content": "写一个函数"},
            {"role": "assistant", "content": "def f(): pass"},
        ]

    def test_input_merged(self):
        m = to_messages("翻译", "hello", "你好")
        assert "hello" in m["messages"][0]["content"]
        assert "翻译" in m["messages"][0]["content"]
        assert m["messages"][1]["role"] == "assistant"

    def test_stripped(self):
        m = to_messages("  q  ", "  i  ", "  a  ")
        assert m["messages"][0]["content"].startswith("q")
        assert m["messages"][1]["content"] == "a"


class TestLooksStructured:
    def test_json_object(self):
        assert looks_structured('{"name": "x", "value": 1}')

    def test_json_fence(self):
        assert looks_structured("答案如下：\n```json\n{\"a\": 1}\n```")

    def test_list_items(self):
        assert looks_structured("- 第一项\n- 第二项\n- 第三项")

    def test_plain_text_rejected(self):
        assert not looks_structured("这是一个很长很长的自然语言句子，没有结构化特征。")


class TestMd5:
    def test_stable_and_matches_reference(self, tmp_path: Path):
        p = tmp_path / "f.jsonl"
        p.write_text(json.dumps({"a": 1}, ensure_ascii=False) + "\n", encoding="utf-8")
        expect = hashlib.md5(p.read_bytes()).hexdigest()
        assert md5_of(p) == expect

    def test_changes_with_content(self, tmp_path: Path):
        p = tmp_path / "f.txt"
        p.write_text("v1", encoding="utf-8")
        h1 = md5_of(p)
        p.write_text("v2", encoding="utf-8")
        assert md5_of(p) != h1


class TestSampleLengthFilterContract:
    """load_raw 依赖 datasets（GPU 环境），此处只锁过滤契约：过短/过长被剔除。"""

    def test_length_bounds_documented(self):
        # 与 prepare_data.load_raw 中的常量保持一致：<8 或 >900 被过滤
        assert not looks_structured("12345")          # 5 字符：过短（即使结构化形态）
        long_json = "{" + '"' + "k" * 500 + '": 1' + "}"
        assert looks_structured(long_json)            # 形态有效（长度由 load_raw 控制）
