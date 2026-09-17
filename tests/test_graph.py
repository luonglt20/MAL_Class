import unittest

from malware_hybrid.graph import ContextualGraphPruner, ExecutionGraphBuilder
from malware_hybrid.schema import ExecutionEvent


def event(index: int, api: str, target: str, success: bool = True) -> ExecutionEvent:
    return ExecutionEvent(
        index=index,
        timestamp=float(index),
        process_id="10",
        parent_process_id="1",
        thread_id="20",
        api=api,
        category="filesystem",
        success=success,
        return_value="0",
        subject="pid:10",
        target=target,
        resource_type="file",
    )


class GraphTests(unittest.TestCase):
    def test_parent_process_relation_is_not_duplicated_per_event(self):
        graph = ExecutionGraphBuilder().build([
            event(0, "OpenProcess", "pid:40"),
            event(1, "WriteProcessMemory", "pid:40"),
        ])
        parent_edges = [edge for edge in graph.edges if edge.edge_type == "parent_of"]
        self.assertEqual(len(parent_edges), 1)

    def test_pruner_only_removes_successful_low_information_system_reads(self):
        graph = ExecutionGraphBuilder().build([
            event(0, "ReadFile", "<SYSTEM32>/kernel32.dll"),
            event(1, "WriteFile", "<SYSTEM32>/changed.dll"),
            event(2, "ReadFile", "<SYSTEM32>/denied.dll", success=False),
        ])
        pruned = ContextualGraphPruner().prune(graph)
        edge_types = [edge.edge_type for edge in pruned.edges]
        self.assertIn("write", edge_types)
        self.assertIn("read", edge_types)  # failed reads remain negative evidence
        self.assertLess(len(pruned.edges), len(graph.edges))


if __name__ == "__main__":
    unittest.main()
