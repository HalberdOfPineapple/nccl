"""Axis limits that show absolute effect size instead of magnifying small changes."""
from matplotlib.ticker import MaxNLocator


def zero_based_upper(values):
    """Leave headroom and use a rounded bound; share it across comparable panels."""
    maximum = max(values, default=0)
    if maximum <= 0:
        return 1
    return MaxNLocator(nbins=5).tick_values(0, maximum * 1.1)[-1]


def set_zero_based_y(axis, upper):
    axis.set_ylim(0, upper)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.ticklabel_format(axis='y', style='plain', useOffset=False)
