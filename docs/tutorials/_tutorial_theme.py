"""Shared plotnine theme for PyRenew tutorials."""

import plotnine as p9

theme_tutorial = p9.theme_bw() + p9.theme(
    text=p9.element_text(size=14),
    line=p9.element_line(size=0.25),
    plot_title=p9.element_text(
        size=16,
        weight="bold",
        linespacing=1.5,
        margin={"t": 6, "b": 6, "l": 2, "r": 2},
    ),
    plot_caption=p9.element_text(size=11),
    axis_title=p9.element_text(size=12),
    axis_text=p9.element_text(size=11),
    legend_title=p9.element_text(size=12),
    legend_text=p9.element_text(size=11),
    strip_text=p9.element_text(size=12),
    strip_background=p9.element_rect(fill="#D9D9D9", color=None),
    figure_size=(10, 6),
)
