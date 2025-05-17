import ast
import textwrap
import types
import unittest


def load_build_graph():
    with open("claim_irregularity_finder.py", "r", encoding="utf-8") as f:
        source = f.read()
    module = ast.parse(source)
    code_sections = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_graph":
            code_sections.append(
                textwrap.dedent("".join(source.splitlines(True)[node.lineno - 1: node.end_lineno]))
            )
    namespace = {}
    class DiGraph:
        def __init__(self):
            self._edges = []
            self._nodes = {}

        def add_node(self, node, **attrs):
            self._nodes[node] = attrs

        def add_edge(self, u, v):
            self._edges.append((u, v))

        def edges(self):
            return self._edges

    namespace['nx'] = types.SimpleNamespace(DiGraph=DiGraph)
    exec(
        "from datetime import datetime\nfrom typing import List, Dict\n" "def tqdm(x, **k):\n    return x\n" + "\n".join(code_sections),
        namespace,
    )
    return namespace["build_graph"]


build_graph = load_build_graph()


class BuildGraphTest(unittest.TestCase):
    def make_events(self):
        return [
            {"id": "e1", "timestamp": "2024-06-01T00:00:00", "actor": "A", "action": "a1", "type": "Action", "details": ""},
            {"id": "e2", "timestamp": "2024-06-02T00:00:00", "actor": "B", "action": "a2", "type": "Action", "details": ""},
            {"id": "e3", "timestamp": "2024-06-03T00:00:00", "actor": "C", "action": "a3", "type": "Action", "details": ""},
        ]

    def test_default_links_immediate(self):
        G = build_graph(self.make_events())
        self.assertEqual(list(G.edges()), [("e1", "e2"), ("e2", "e3")])

    def test_max_back_links(self):
        G = build_graph(self.make_events(), max_back_links=2)
        self.assertEqual(set(G.edges()), {("e1", "e2"), ("e2", "e3"), ("e1", "e3")})

    def test_explicit_reference(self):
        events = self.make_events()
        events[2]["details"] = "In response to earlier issues"
        G = build_graph(events)
        self.assertEqual(set(G.edges()), {("e1", "e2"), ("e1", "e3"), ("e2", "e3")})


if __name__ == "__main__":
    unittest.main()
