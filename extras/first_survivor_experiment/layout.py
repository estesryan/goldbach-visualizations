"""Figure sizes and panel geometry for the first-survivor figures.

All rectangles are (x, y, w, h) in inches. Cards are measured from the figure's
lower-left corner; the axes inside a card from the card's lower-left corner. The
portrait layout is designed on its own, not mapped from the landscape one: its
cards are full width, so the two zooms of panel A sit side by side over the
overview strip, and panels B and C put their charts side by side.
"""

FIGURE_SIZES = {
    "landscape": (16.0, 9.0),           # 16:9, 3200 x 1800 px
    "preview_4x5": (16.2, 20.25),       # 4:5, 2160 x 2700 px, as the poster
}
EXPORT_WIDTH_PX = {"landscape": 3200, "preview_4x5": 2160}

BLOCKS_SIZE = (10.0, 8.0)               # the appendix figure
BLOCKS_WIDTH_PX = 2000

MARGIN, GAP = 0.144, 0.09
TEXT_X = 0.216                          # left edge of header and footer text

HEADER_IN = {"landscape": 1.42, "preview_4x5": 1.62}
FOOTER_IN = {"landscape": 1.12, "preview_4x5": 1.12}
# Characters per line of the header (body size) and footer (small size) text.
HEADER_WRAP = {"landscape": 196, "preview_4x5": 198}
FOOTER_WRAP = {"landscape": 220, "preview_4x5": 222}


def cards(name):
    """The three cards "example", "boundary" and "envelope"."""
    W, H = FIGURE_SIZES[name]
    y0, h = FOOTER_IN[name], H - HEADER_IN[name] - FOOTER_IN[name]
    names = ("example", "boundary", "envelope")
    if W > H:
        w = (W - 2 * MARGIN - 2 * GAP) / 3
        return {k: (MARGIN + i * (w + GAP), y0, w, h) for i, k in enumerate(names)}
    heights = {"example": 4.65, "boundary": 6.25}
    heights["envelope"] = h - heights["example"] - heights["boundary"] - 2 * GAP
    out, y = {}, y0 + h
    for k in names:
        y -= heights[k]
        out[k] = (MARGIN, y, W - 2 * MARGIN, heights[k])
        y -= GAP
    return out


def panel_axes(name):
    """Axes rectangles inside each card, keyed by card and then by axes name."""
    c = cards(name)
    if name == "landscape":
        w = c["example"][2]
        return {
            "example": {"centre": (0.38, 3.72, w - 0.7, 1.2),
                        "overview": (0.38, 2.62, w - 0.7, 0.42),
                        "edge": (0.38, 0.98, w - 0.7, 1.08),
                        "legend": (0.25, 0.12, w - 0.4, 0.42)},
            "boundary": {"small": (0.72, 3.15, w - 0.95, 1.55),
                         "large": (0.72, 0.62, w - 1.62, 1.76),
                         "colorbar": (w - 0.78, 0.62, 0.08, 1.76)},
            "envelope": {"main": (0.78, 2.78, w - 1.02, 2.3),
                         "drift": (0.78, 0.6, w - 1.82, 1.32)},
        }
    w = c["example"][2]
    half = (w - 1.1) / 2
    return {
        "example": {"centre": (0.38, 2.2, half - 0.2, 1.2),
                    "edge": (0.38 + half + 0.55, 2.2, half - 0.2, 1.2),
                    "overview": (0.38, 0.72, w - 0.76, 0.4),
                    "legend": (0.25, 0.06, w - 0.5, 0.32)},
        "boundary": {"small": (0.8, 0.68, 7.2, 4.1),
                     "large": (8.85, 0.68, 6.05, 4.1),
                     "colorbar": (15.05, 0.68, 0.1, 4.1)},
        "envelope": {"main": (0.85, 0.68, 9.7, 4.4),
                     "drift": (11.45, 0.68, 3.55, 3.8)},
    }
