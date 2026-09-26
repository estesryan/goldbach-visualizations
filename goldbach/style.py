"""Palette, semantic colour roles, colormap, fonts and matplotlib settings.

Importing this module changes nothing global. Call apply_style() before
creating figures.
"""
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba

# Colours used throughout the visualizations.
PALETTE = {
    "BG":        "#0f0e0d",   # page background
    "PANEL_BG":  "#171513",   # panel fill
    "BORDER":    "#2c2825",   # panel borders, spines
    "GRID":      "#2a2622",   # gridlines, wheel spokes
    "TEXT":      "#f1ebe2",   # primary text
    "MUTED":     "#aaa39c",   # secondary text (7.3:1 on PANEL_BG)
    "EXCLUDED":  "#4f4943",   # composite / excluded classes, background lattice
    "ACCENT":    "#ff6b35",   # prime / admissible / surviving
    "ACCENT_2":  "#7fb3c8",   # partner q = N - p
    "HIGHLIGHT": "#fff3d6",   # Goldbach pair links
    "WARNING":   "#c9a54c",   # small-prime exceptions
}

# Semantic role -> palette key. Artists carry their role as gid.
SEMANTIC = {
    "prime": "ACCENT", "admissible": "ACCENT", "surviving": "ACCENT",
    "partner": "ACCENT_2",
    "excluded": "EXCLUDED", "lattice": "EXCLUDED",
    "pair": "HIGHLIGHT",
    "exception": "WARNING",
    "fixed": "TEXT",
}


def C(role):
    """Colour for a semantic role or a palette key."""
    return PALETTE[SEMANTIC.get(role, role)]


def mix(a, b, t):
    """Blend a -> b by t. Only used to build the colormap."""
    ra, rb = np.array(to_rgba(a)), np.array(to_rgba(b))
    return tuple((1 - t) * ra + t * rb)


# Sequential colormap for counts: PANEL_BG -> dim ACCENT -> ACCENT -> HIGHLIGHT.
# Masked (zero) cells show as BG.
BRAND_CMAP = LinearSegmentedColormap.from_list("brand", [
    PALETTE["PANEL_BG"], mix(PALETTE["PANEL_BG"], PALETTE["ACCENT"], 0.35),
    PALETTE["ACCENT"], PALETTE["HIGHLIGHT"]])
BRAND_CMAP.set_bad(PALETTE["BG"])

FONT_BODY = ["Poppins", "DejaVu Sans"]          # DejaVu covers ≡, ↦, φ, ω per glyph
FONT_HEAD = ["Lora", "DejaVu Serif"]
FS = {"title": 22, "head": 17, "num": 12, "sub": 11, "body": 11, "small": 10, "tick": 9.5}

RC_PARAMS = {
    "font.family": FONT_BODY, "font.size": FS["body"],
    "text.color": PALETTE["TEXT"], "axes.labelcolor": PALETTE["MUTED"],
    "xtick.color": PALETTE["MUTED"], "ytick.color": PALETTE["MUTED"],
    "xtick.labelcolor": PALETTE["MUTED"], "ytick.labelcolor": PALETTE["MUTED"],
    "axes.edgecolor": PALETTE["BORDER"], "axes.facecolor": PALETTE["PANEL_BG"],
    "figure.facecolor": PALETTE["BG"], "savefig.facecolor": PALETTE["BG"],
    "axes.labelsize": FS["small"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "grid.color": PALETTE["GRID"], "lines.color": PALETTE["TEXT"],
    "patch.edgecolor": PALETTE["BORDER"], "patch.facecolor": PALETTE["PANEL_BG"],
    "legend.edgecolor": PALETTE["BORDER"], "legend.facecolor": PALETTE["PANEL_BG"],
    "legend.labelcolor": PALETTE["TEXT"], "legend.fontsize": FS["small"],
}


def apply_style():
    """Set matplotlib's global rcParams for these figures."""
    matplotlib.rcParams.update(RC_PARAMS)


def contrast(a, b):
    """WCAG contrast ratio of two colours."""
    def lum(c):
        v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
