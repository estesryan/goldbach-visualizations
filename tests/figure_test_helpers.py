"""Figure-test helpers shared by the poster, parity and first-survivor figure tests.

They read drawn artists back from a matplotlib figure: text, scatter offsets, the
colours each artist draws with, and text contrast.
"""
from collections import defaultdict

import matplotlib
import numpy as np
from matplotlib.collections import Collection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text

from goldbach.style import C, PALETTE, SEMANTIC, contrast

MIN_TEXT_CONTRAST = 4.5     # WCAG AA


def all_text(fig):
    return "\n".join(t.get_text() for t in fig.findobj(Text) if t.get_visible())


def points(ax, gid, hollow=None):
    """Offsets of one role's scatter points, rounded. hollow=True: unfilled markers only."""
    out = set()
    for c in ax.collections:
        if c.get_gid() != gid:
            continue
        fc = c.get_facecolor()
        is_hollow = len(fc) == 0 or to_rgba(fc[0])[3] == 0
        if hollow is None or hollow == is_hollow:
            out |= {(round(float(x), 6), round(float(y), 6)) for x, y in c.get_offsets()}
    return out


def rounded(xy):
    return {(round(float(x), 6), round(float(y), 6)) for x, y in xy}


def artist_colours(art):
    """Non-transparent colours an artist draws with. Colormapped collections are skipped."""
    out = []
    if isinstance(art, Line2D):
        if art.get_linestyle() not in ("None", "", " ") and art.get_linewidth() > 0:
            out.append(art.get_color())
        if art.get_marker() not in (None, "None", "", " "):
            out += [art.get_markerfacecolor(), art.get_markeredgecolor()]
    elif isinstance(art, Collection):
        if art.get_array() is not None:
            return []
        out += list(art.get_facecolor())
        if np.any(art.get_linewidths()):
            out += list(art.get_edgecolor())
    elif isinstance(art, matplotlib.patches.Patch):
        out.append(art.get_facecolor())
        if art.get_linewidth() > 0:
            out.append(art.get_edgecolor())
    elif isinstance(art, Text):
        if art.get_text().strip():
            out.append(art.get_color())
    return [to_hex(c) for c in out if c is not None and to_rgba(c)[3] > 0]


def stray_colours(fig):
    """Colours drawn that are not in PALETTE."""
    allowed = {v.lower() for v in PALETTE.values()}
    return [(type(a).__name__, c) for a in fig.findobj() if a.get_visible()
            for c in artist_colours(a) if c.lower() not in allowed]


def assert_semantic_roles(fig):
    role_cols = defaultdict(set)
    for a in fig.findobj():
        if a.get_visible() and a.get_gid() in SEMANTIC:
            role_cols[a.get_gid()] |= {c.lower() for c in artist_colours(a)}
    assert role_cols
    for role, cols in role_cols.items():
        assert C(role).lower() in cols, role
        # Only the role's own colour, plus the panel fill and border used as edges.
        assert cols <= {C(role).lower(), C("PANEL_BG").lower(), C("BORDER").lower()}, role


def low_contrast_text(fig, background=lambda t: C("PANEL_BG")):
    """Visible text below MIN_TEXT_CONTRAST against its background (PANEL_BG by default)."""
    low = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bg = background(t)
        if contrast(t.get_color(), bg) < MIN_TEXT_CONTRAST:
            low.append((t.get_text()[:30], round(contrast(t.get_color(), bg), 2)))
    return low


def written_files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
