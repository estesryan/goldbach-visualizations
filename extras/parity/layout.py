"""Figure sizes and card layout for the parity visualization.

Three cards between a header (title and introduction) and a footer (caption): side
by side in landscape, stacked in the 4:5 layout. Sizes in inches; the 4:5 size is the
poster's.
"""

FIGURE_SIZES = {
    "landscape": (16.0, 9.0),           # 16:9
    "preview_4x5": (16.2, 20.25),       # 4:5, as the poster
}
EXPORT_WIDTH_PX = {"landscape": 3200, "preview_4x5": 2160}
HEADER_IN = 1.40            # title and introduction, above the cards
FOOTER_IN = 1.30            # caption, below the cards


def cards(W, H):
    """The three cards (x, y, w, h) in inches: "average" (S/M for each N), "sums"
    (M and S for the featured N) and "classes" (sign splits of groups of pairs).
    Side by side in landscape, stacked in the 4:5 layout."""
    m, g = 0.144, 0.09
    y0, h = FOOTER_IN, H - HEADER_IN - FOOTER_IN
    names = ("average", "sums", "classes")
    if W > H:
        w = (W - 2 * m - 2 * g) / 3
        return {k: (m + i * (w + g), y0, w, h) for i, k in enumerate(names)}
    h = (h - 2 * g) / 3
    return {k: (m, y0 + (2 - i) * (h + g), W - 2 * m, h) for i, k in enumerate(names)}


LAYOUTS = {name: cards(*size) for name, size in FIGURE_SIZES.items()}
