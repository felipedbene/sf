import ast
import textwrap
import types


def load_parse_events():
    with open("claim_irregularity_finder.py", "r", encoding="utf-8") as f:
        source = f.read()
    module = ast.parse(source)
    code_sections = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "parse_events",
            "_clean_text",
            "_normalize_timestamp",
        ):
            code_sections.append(
                textwrap.dedent("".join(source.splitlines(True)[node.lineno - 1 : node.end_lineno]))
            )
    namespace = {}
    exec(
        "import re\nfrom typing import List, Dict\nfrom datetime import datetime\n"
        "def tqdm(x, **k):\n    return x\n" + "\n".join(code_sections),
        namespace,
    )
    return namespace["parse_events"]


parse_events = load_parse_events()
import unittest


class ParseEventsTest(unittest.TestCase):
    def test_blank_lines(self):
        text = (
            "2024-06-15 14:10 Actor1: Did something\n\n"
            "2024-06-16 09:00 Actor2: Did another"
        )
        events = parse_events([text])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["actor"], "Actor1")
        self.assertEqual(events[1]["actor"], "Actor2")


    def test_crlf_and_blank_lines(self):
        text = (
            "2024-06-15 14:10 Actor1: Did something\r\n\r\n"
            "2024-06-16 09:00 Actor2: Did another"
        )
        events = parse_events([text])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["actor"], "Actor1")
        self.assertEqual(events[1]["actor"], "Actor2")


if __name__ == "__main__":
    unittest.main()
