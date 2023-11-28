import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def hide_spines(ax: plt.Axes):
    """
    Hides the lines around the graph.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)

def annotate_bars(
    ax: plt.Axes,
    lim: int = 10,
    percentage: bool = False,
    offset_ratio: float = 0.05,
    total: float = None,
    orientation: str = "vertical",
):
    if total is None:
        if orientation == "vertical":
            total = sum([p.get_height() for p in ax.patches])
        else:
            total = sum([p.get_width() for p in ax.patches])

    for bar in ax.patches:
        if orientation == "vertical":
            height = bar.get_height()
            offset = height * offset_ratio
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height - offset if (height / total) * 100 > lim else height + offset,
                f"{(height / total) * 100:.2f}%" if percentage else str(int(height)),
                ha="center",
                color="w" if (height / total) * 100 > lim else "black",
                fontsize=11,
            )
        else:  # horizontal
            width = bar.get_width()
            offset = width * offset_ratio
            ax.text(
                width - offset if (width / total) * 100 > lim else width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{(width / total) * 100:.2f}%" if percentage else str(int(width)),
                va="center",
                color="w" if (width / total) * 100 > lim else "black",
                fontsize=11,
            )

