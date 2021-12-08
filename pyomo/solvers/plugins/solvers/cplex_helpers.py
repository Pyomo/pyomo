import re
from typing import Optional, List


def get_root_node_processing_time(log_output: str) -> Optional[float]:
    # use regular expressions to use multi-line match patterns:
    root_node_processing_time = re.search(
        r"Root node processing.*:\n\s+Real time\s+=\s+(\d+\.\d+) sec", log_output
    )
    if root_node_processing_time:
        return float(root_node_processing_time.group(1))
    return None


def get_tree_processing_time(log_output: str) -> Optional[float]:
    tree_processing_time = re.search(
        r"(?:Parallel|Sequential).*\n\s+Real time\s+=\s+(\d+\.\d+) sec", log_output
    )
    if tree_processing_time:
        return float(tree_processing_time.group(1))
    return None


