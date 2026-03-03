"""Make plots comparing performance for each label and sample on all available data (using fractional experiments)."""

import functools
from pathlib import Path

import pandas as pd

from cryovit.types import Sample
from cryovit.visualization.utils import (
    compute_stats,
    merge_experiments,
    significance_test,
)


def _plot_df(
    df: pd.DataFrame,
    pvalues: pd.Series,
    key: str,
    title: str,
    file_name: str,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator

    matplotlib.use("Agg")
    colors = sns.color_palette("deep")[:3]
    sns.set_theme(style="darkgrid", font="Open Sans")

    hue_palette = {
        "3D U-Net": colors[0],
        "CryoViT": colors[1],
        "SAM2": colors[2],
    }

    sample_counts = df["sample"].value_counts()
    num_models = df[key].nunique()
    sorted_samples = sample_counts.sort_values(ascending=True).index.tolist()
    fig = plt.figure(figsize=(20, 6))
    ax = plt.gca()

    params = {
        "x": "sample",
        "y": "dice_metric",
        "hue": key,
        "data": df,
        "order": sorted_samples,
    }

    sns.boxplot(
        showfliers=False,
        palette=hue_palette,
        linewidth=1,
        medianprops={"linewidth": 2, "color": "firebrick"},
        ax=ax,
        **params,
    )
    sns.stripplot(
        dodge=True,
        marker=".",
        alpha=0.5,
        palette="dark:black",
        ax=ax,
        **params,
    )

    k1, k2, k3 = df[key].unique()
    pairs = (
        [[(s, k1), (s, k2)] for s in pvalues.index.unique()]
        + [[(s, k1), (s, k3)] for s in pvalues.index.unique()]
        + [[(s, k2), (s, k3)] for s in pvalues.index.unique()]
    )

    annotator = Annotator(ax, pairs, **params)
    annotator.configure(color="blue", line_width=1, verbose=False)
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    new_labels = [
        f"{Sample[label.get_text()]}\n(n={sample_counts[label.get_text()] // num_models})"
        for label in current_labels
    ]

    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, ha="center")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.suptitle(title)
    fig.supxlabel("Sample Name (Count)")
    fig.supylabel("Dice Score")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], loc="lower center", shadow=True)

    plt.tight_layout(rect=(0.01, 0.01, 1.0, 1.0))
    plt.savefig(f"{file_name}.svg")
    plt.savefig(f"{file_name}.png", dpi=300)


def process_multi_label_sample_experiment(
    exp_type: str,
    exp_names: dict[str, list[str]],
    exp_dir: Path,
    result_dir: Path,
):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        exp_type (str): Type of experiment, i.e., "multi_label_sample"
        exp_names (dict[str, list[str]]): Dictionary mapping experiment names to model used and label
        exp_dir (Path): Directory containing the experiment results
        result_dir (Path): Directory to save the results
    """

    result_dir.mkdir(parents=True, exist_ok=True)
    df = merge_experiments(exp_dir, exp_names, keys=["model", "label"])
    df = df[df["split_id"] == 10]  # Use data from 100% of training data

    labels = df["label"].unique()
    for label in labels:
        print(f"Processing label: {label}")
        df_label = df[df["label"] == label]

        total_ps = None
        for m1, m2 in [
            ("CryoViT", "3D U-Net"),
            ("CryoViT", "SAM2"),
            ("3D U-Net", "SAM2"),
        ]:
            test_fn = functools.partial(
                significance_test,
                model_A=m1,
                model_B=m2,
                key="model",
                test_fn="ttest_rel",
            )
            p_values = compute_stats(
                df_label,
                group_keys=["sample", "model"],
                file_name=str(
                    result_dir / f"{exp_type}_{label}_{m1}_{m2}_stats.csv"
                ),
                test_fn=test_fn,
            )
            total_ps = (
                p_values
                if total_ps is None
                else pd.concat([total_ps, p_values])
            )
        assert total_ps is not None

        full_labels = {
            "mito": "Mitochondria",
            "cristae": "Cristae",
            "microtubule": "Microtubules",
            "granule": "Granules",
            "bacteria": "Bacteria",
            "mito_membrane": "Mitochondrial Membrane",
            "er_membrane": "ER Membrane",
        }

        _plot_df(
            df_label,
            total_ps,
            "model",
            f"Model Comparison per Sample for {full_labels[label]}",
            str(result_dir / f"{exp_type}_{label}_comparison"),
        )
