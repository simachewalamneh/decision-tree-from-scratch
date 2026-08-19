
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _impurity_of(node):
    """Works for both Node.impurity_value and RegressionNode.mse_value."""
    return getattr(node, "impurity_value", None) or getattr(node, "mse_value", 0.0)


def _impurity_label(node):
    if hasattr(node, "impurity_value"):
        return f"impurity={node.impurity_value:.3f}"
    return f"mse={node.mse_value:.3f}"


def _prediction_label(node):
    pred = node.prediction
    if isinstance(pred, float):
        return f"predict={pred:.2f}"
    return f"predict={pred}"


def _question_label(node, feature_names=None):
    name = feature_names[node.feature_index] if feature_names else f"x[{node.feature_index}]"
    if node.feature_type == "numerical":
        return f"{name} <= {node.threshold:.2f}?"
    return f"{name} == {node.category!r}?"


def _assign_positions(node, depth=0, x_counter=None, positions=None):
    """Recursively assign (x, y) positions -- leaves get sequential x slots
    left to right, internal nodes are centered above their children."""
    if x_counter is None:
        x_counter = [0]
    if positions is None:
        positions = {}

    if node.is_leaf:
        x = x_counter[0]
        x_counter[0] += 1
        positions[id(node)] = (x, -depth)
        return positions[id(node)]

    left_pos = _assign_positions(node.left, depth + 1, x_counter, positions)
    right_pos = _assign_positions(node.right, depth + 1, x_counter, positions)
    x = (left_pos[0] + right_pos[0]) / 2.0
    positions[id(node)] = (x, -depth)
    return positions[id(node)]


def plot_tree(root, feature_names=None, title=None, figsize=None):
 
    positions = {}
    _assign_positions(root, positions=positions)
    max_x = max(p[0] for p in positions.values()) or 1
    max_depth = max(-p[1] for p in positions.values()) or 1

    if figsize is None:
        figsize = (max(6, max_x * 1.6), max(4, max_depth * 1.6))
    fig, ax = plt.subplots(figsize=figsize)

    box_w, box_h = 0.85, 0.6

    def draw(node):
        x, y = positions[id(node)]
        is_leaf = node.is_leaf

        if is_leaf:
            facecolor = "#DCEEDD"
            edgecolor = "#4C9A5B"
            label = f"{_prediction_label(node)}\nn={node.n_samples}\n{_impurity_label(node)}"
        else:
            facecolor = "#E7EEF7"
            edgecolor = "#3F6D6D"
            label = f"{_question_label(node, feature_names)}\nn={node.n_samples}\n{_impurity_label(node)}"

        box = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.3, edgecolor=edgecolor, facecolor=facecolor, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=4)

        if not is_leaf:
            for child, branch_label in [(node.left, "Yes"), (node.right, "No")]:
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [y - box_h / 2, cy + box_h / 2],
                        color="#E0D3D3", linewidth=1.0, zorder=1)
                mx, my = (x + cx) / 2, (y + cy) / 2
                ax.text(mx, my, branch_label, fontsize=7, color="#555555",
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                    edgecolor="none", alpha=0.85), zorder=2)
                draw(child)

    draw(root)

    ax.set_xlim(-0.7, max_x + 0.7)
    ax.set_ylim(-max_depth - 0.7, 0.7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig
