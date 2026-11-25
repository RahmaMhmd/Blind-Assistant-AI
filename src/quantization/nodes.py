def _node_attrs(node) -> tuple[int, list[int]]:
    attrs = {a.name: a for a in node.attribute}
    group = int(attrs["group"].i) if "group" in attrs else 1
    kernel = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else []
    return group, kernel


def select_conv_nodes(graph, strategy: str, tail_count: int = 13) -> list[str]:
    """pick safe conv layers to quantize"""
    convs = [n for n in graph.node if n.op_type == "Conv"]

    if strategy == "conv_all":
        return [n.name for n in convs]

    if strategy == "backbone_tail":
        backbone = [n.name for n in convs if "/backbone/" in n.name.lower()]
        return backbone[-tail_count:] if tail_count > 0 else backbone

    if strategy == "dfine_mixed":
        stage3, stage2_1x1 = [], []
        for node in convs:
            low = node.name.lower()
            group, kernel = _node_attrs(node)
            if "/backbone/" in low and group == 1:
                if "stages.3" in low:
                    stage3.append(node.name)
                elif "stages.2" in low and kernel == [1, 1]:
                    stage2_1x1.append(node.name)
        return stage3[:-1] + stage2_1x1

    raise ValueError(f"unknown quantization strategy: {strategy}")
