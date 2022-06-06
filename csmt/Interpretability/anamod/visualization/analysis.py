"""Visualize model analysis results"""

import codecs
from types import SimpleNamespace

import anytree
from anytree.exporter import DotExporter
import matplotlib.patches as patches  # pylint: disable = consider-using-from-import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from anamod.core import constants


def visualize_temporal(args, features, sequence_length):
    """Visualize temporal feature importance results"""
    # pylint: disable = invalid-name, too-many-locals
    features = list(filter(lambda feature: feature.important, features))  # Filter out unimportant features
    num_features = len(features)
    if num_features == 0:
        print("No important features identified, skipping window feature importance window visualization.")
        return
    sns.set(rc={'figure.figsize': (sequence_length / 2, 2 + num_features / 2), 'figure.dpi': 300,
                'font.family': 'Serif', 'font.serif': 'Palatino',
                'axes.titlesize': 24, 'axes.labelsize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 24})
    features = sorted(features, key=lambda feature: feature.window_effect_size, reverse=True)
    data = np.zeros((num_features, sequence_length))
    hatchdata = np.zeros((num_features, sequence_length))
    labels = [feature.name for feature in features]
    for idx, feature in enumerate(features):
        if feature.temporal_window:
            left, right = feature.temporal_window
            data[idx, left: right + 1] = feature.window_effect_size
        else:
            data[idx] = feature.overall_effect_size
        if feature.window_ordering_important:
            hatchdata[idx, left: right + 1] = 1
    _, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [num_features, 1]})
    sns.heatmap(data, yticklabels=labels, xticklabels=np.arange(1, sequence_length + 1),
                mask=(data == 0), linewidth=2, linecolor="black", cmap="YlOrRd",
                cbar_kws=dict(label="Importance\nScore"), ax=axes[0])
    # Border lines
    axes[0].axhline(y=0, color='black', linewidth=4)
    axes[0].axhline(y=num_features, color='black', linewidth=4)
    axes[0].axvline(x=0, color='black', linewidth=4)
    axes[0].axvline(x=sequence_length, color='black', linewidth=4)
    # Labels
    # axes[0].set_title("Temporal Feature Importance")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Important\nFeatures")
    # Hatch texture for ordering relevance
    x = np.arange(sequence_length + 1)
    y = np.arange(num_features + 1)
    z = np.ma.masked_equal(hatchdata, 0)
    axes[0].pcolor(x, y, z, hatch='//', alpha=0.)
    # Grey foreground for non-relevant timesteps
    z = np.ma.masked_not_equal(data, 0)
    axes[0].pcolor(x, y, z, cmap="Greys", linewidth=2, edgecolors="Grey")
    # Fix edges of relevant timesteps
    z = np.ma.masked_equal((data != 0), 0)
    axes[0].pcolor(x, y, z, linewidth=2, edgecolor="k", facecolor="none", alpha=1.0)
    output_filename = f"{args.output_dir}/{constants.FEATURE_IMPORTANCE_WINDOWS}.png"
    # Legend
    axes[1].set_aspect(1, anchor="W")
    axes[1].add_patch(patches.Rectangle((0, 0), 1, 1, facecolor="w", edgecolor="k", linewidth=4, hatch="//"))
    axes[1].annotate("Ordering important", (1, 0.4), color="k", weight="bold", fontsize=16)
    axes[1].xaxis.set_ticklabels([])
    axes[1].yaxis.set_ticklabels([])
    # Plot
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Feature importance visualization: {output_filename}")


def visualize_hierarchical(args, features):
    """Visualize hierarchical feature importance results"""
    tree = features[0].root
    opts = SimpleNamespace(output_dir=args.output_dir, effect_name="Importance Score", color_scheme="ylorrd9",
                           color_range=[1, 9], sorting_param=constants.EFFECT_SIZE, minimal_labels=False, rectangle_leaves=True)
    nodes = {}
    for node in anytree.LevelOrderIter(tree):
        if node.important or node.name == constants.DUMMY_ROOT:
            parent = nodes[node.parent.name] if node.parent else None
            newnode = anytree.Node(node.name, parent=parent,
                                   description=node.description, effect_size=node.effect_size, was_leaf=node.is_leaf)
            nodes[newnode.name] = newnode
    if len(nodes) <= 1:
        print("No important features identified, skipping feature importance hierarchy visualization.")
        return
    newtree = next(iter(nodes.values())).root  # identify root
    if newtree.name == constants.DUMMY_ROOT and len(newtree.children) == 1:
        # Get rid of dummy node if not required to maintain tree
        newtree = newtree.children[0]
        newtree.parent = None
    color_nodes(opts, newtree)
    render_tree(opts, newtree)


def render_tree(opts, tree):
    """Render tree in ASCII and graphviz"""
    with codecs.open(f"{opts.output_dir}/{constants.FEATURE_IMPORTANCE_HIERARCHY}.txt", "w", encoding="utf8") as txt_file:
        for pre, _, node in anytree.RenderTree(tree):
            txt_file.write(f"{pre}{node.name}: {node.description.title()} ({opts.effect_name}: {str(node.effect_size)})\n")
    graph_options = []  # Example: graph_options = ["dpi=300.0;", "style=filled;", "bgcolor=yellow;"]
    exporter = DotExporter(tree, options=graph_options, nodeattrfunc=lambda node: nodeattrfunc(opts, node))
    exporter.to_dotfile(f"{opts.output_dir}/{constants.FEATURE_IMPORTANCE_HIERARCHY}.dot")
    try:
        output_filename = f"{opts.output_dir}/{constants.FEATURE_IMPORTANCE_HIERARCHY}.png"
        exporter.to_picture(output_filename)
        print(f"Feature importance visualization: {output_filename}")
    except FileNotFoundError:
        print("Feature importance visualization: error during tree rendering. Is Graphviz installed on your system?")


def color_nodes(opts, tree):
    """Add fill and font color to nodes based on partition in sorted list"""
    nodes_sorted = sorted(anytree.LevelOrderIter(tree), key=lambda node: node.effect_size)  # sort nodes for color grading
    num_nodes = len(nodes_sorted)
    lower, upper = opts.color_range
    num_colors = upper - lower + 1
    assert 1 <= lower <= upper <= 9
    for idx, node in enumerate(nodes_sorted):
        node = nodes_sorted[idx]
        node.color = idx + lower
        if num_nodes > num_colors:
            node.color = lower + (idx * num_colors) // num_nodes
        assert node.color in range(lower, upper + 1, 1)
    # Non-differentiated nodes should have the same color
    prev_node = None
    for node in nodes_sorted:
        if prev_node and node.effect_size == prev_node.effect_size:
            node.color = prev_node.color
        prev_node = node
        node.fontcolor = "black" if node.color <= 5 else "white"


def nodeattrfunc(opts, node):
    """Node attributes function"""
    label = node.name.upper()
    if not opts.minimal_labels and node.description:
        label = f"{label}:\n{node.description}"
    words = label.split(" ")
    words_per_line = 3
    lines = []
    for idx in range(0, len(words), words_per_line):
        line = " ".join(words[idx: min(len(words), idx + words_per_line)])
        lines.append(line)
    label = "\n".join(lines)
    if not opts.minimal_labels and node.effect_size:
        label = f"{label}\n{opts.effect_name}: {node.effect_size:0.3f}"
    shape = "rectangle" if opts.rectangle_leaves and node.was_leaf else "ellipse"
    return (f"fillcolor=\"/{opts.color_scheme}/{node.color}\" label=\"{label}\" style=filled "
            f"fontname=\"helvetica bold\" fontsize=15.0 fontcolor={node.fontcolor} shape = {shape}")
