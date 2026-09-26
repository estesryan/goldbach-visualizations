"""Figure sizes, panel rectangles and the coordinate mapping between layouts.

Drawing code is written in landscape figure fractions. For other layouts each
panel's content is mapped into that layout's panel rectangle. The header keeps
a fixed height (HDR_IN inches); the content area below it is rescaled.
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Figure size in inches. Every rectangle below is designed for these sizes.
FIGURE_SIZES = {
    "landscape": (18.0, 12.0),          # 3:2
    "linkedin_4x5": (16.2, 20.25),      # 4:5, LinkedIn portrait 1080 x 1350
}
LAND = FIGURE_SIZES["landscape"]
LAND_RECTS = {1: (0.008, 0.655, 0.326, 0.305), 2: (0.339, 0.655, 0.326, 0.305),
              3: (0.670, 0.655, 0.322, 0.305), 4: (0.008, 0.335, 0.40, 0.312),
              5: (0.413, 0.335, 0.579, 0.312), 6: (0.008, 0.008, 0.44, 0.319),
              7: (0.453, 0.008, 0.539, 0.319)}
HDR_IN = 0.9            # height of panel heading + subtitle zone, inches


def _in(r, size):
    return (r[0] * size[0], r[1] * size[1], r[2] * size[0], r[3] * size[1])


def portrait_rects(W, H):
    """Portrait panel rects in inches. Rows: 1-2, 3-4, 5 (full width, tallest), 6-7."""
    m, g, top = 0.144, 0.09, 0.48
    hero = 5.5
    row = (H - top - m - hero - 3 * g) / 3
    full = W - 2 * m
    half = (full - g) / 2
    r = {}
    y = H - top - row
    r[1], r[2] = (m, y, half, row), (m + half + g, y, half, row)
    y -= g + row
    r[3], r[4] = (m, y, half, row), (m + half + g, y, half, row)
    y -= g + hero
    r[5] = (m, y, full, hero)
    y -= g + row
    w6 = 7.6
    r[6], r[7] = (m, y, w6, row), (m + w6 + g, y, full - w6 - g, row)
    return r


# Layout name -> panel number -> (x, y, w, h) in inches.
LAYOUTS = {
    "landscape": {k: _in(v, LAND) for k, v in LAND_RECTS.items()},
    "linkedin_4x5": portrait_rects(*FIGURE_SIZES["linkedin_4x5"]),
}


class PanelCanvas:
    """One figure in one layout, plus the panel currently being drawn.

    point() maps a landscape point and size() a landscape size into the current
    panel. axes(), text(), line() and rect() add artists at landscape coords.
    """

    def __init__(self, fig, fig_in, rects):
        self.fig = fig
        self.fig_in = fig_in        # figure size in inches
        self.rects = rects          # panel number -> rect in inches for this layout
        self.land = None            # current panel's landscape rect, inches
        self.new = None             # current panel's rect in this layout, inches

    def set_panel(self, num):
        self.land = _in(LAND_RECTS[num], LAND)
        self.new = self.rects[num]

    def point(self, x, y):
        lx, ly, lw, lh = self.land
        nx, ny, nw, nh = self.new
        sx, sy = nw / lw, (nh - HDR_IN) / (lh - HDR_IN)
        X = nx + (x * LAND[0] - lx) * sx
        Y = ny + nh - (HDR_IN + ((ly + lh) - y * LAND[1] - HDR_IN) * sy)
        return X / self.fig_in[0], Y / self.fig_in[1]

    def size(self, w, h):
        lx, ly, lw, lh = self.land
        nx, ny, nw, nh = self.new
        sx, sy = nw / lw, (nh - HDR_IN) / (lh - HDR_IN)
        return w * LAND[0] * sx / self.fig_in[0], h * LAND[1] * sy / self.fig_in[1]

    def axes(self, rect, **kw):
        x, y = self.point(rect[0], rect[1]); w, h = self.size(rect[2], rect[3])
        return self.fig.add_axes([x, y, w, h], **kw)

    def text(self, x, y, s, **kw):
        X, Y = self.point(x, y)
        return self.fig.text(X, Y, s, **kw)

    def line(self, xs, ys, **kw):
        pts = [self.point(x, y) for x, y in zip(xs, ys)]
        ln = Line2D([p[0] for p in pts], [p[1] for p in pts], transform=self.fig.transFigure, **kw)
        self.fig.add_artist(ln)
        return ln

    def rect(self, xy, w, h, **kw):
        X, Y = self.point(*xy); W, H = self.size(w, h)
        rc = Rectangle((X, Y), W, H, transform=self.fig.transFigure, **kw)
        self.fig.patches.append(rc)
        return rc
