from manim import *
import numpy as np
import matplotlib.pyplot as plt


class BasisSpline(object):
    def __init__(
        self,
        n_df,
        xrange=(0, 1),
        k=4,
        knots=None,
        normalize=True,
    ):
        self.order = k
        self.N = n_df
        self.xrange = xrange
        if knots is None:
            interior_knots = np.linspace(*xrange, n_df - k + 2)
            dx = interior_knots[1] - interior_knots[0]
            knots = np.concatenate(
                [
                    xrange[0] - dx * np.arange(1, k)[::-1],
                    interior_knots,
                    xrange[1] + dx * np.arange(1, k),
                ]
            )
        self.knots = knots
        self.interior_knots = knots
        assert len(self.knots) == self.N + self.order

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            grid = np.linspace(*xrange, 1000)
            grid_bases = np.array(self.bases(grid))
            self.basis_vols = np.array(
                [np.trapz(grid_bases[i, :], grid) for i in range(self.N)]
            )

    def norm(self, coefs):
        n = 1.0 / np.sum(self.basis_vols * coefs.flatten()) if self.normalize else 1.0
        return n

    def _basis(self, xs, i, k):
        if self.knots[i + k] - self.knots[i] < 1e-6:
            return np.zeros_like(xs)
        elif k == 1:
            v = np.zeros_like(xs)
            v[(xs >= self.knots[i]) & (xs < self.knots[i + 1])] = 1 / (
                self.knots[i + 1] - self.knots[i]
            )
            return v
        else:
            v = (xs - self.knots[i]) * self._basis(xs, i, k - 1) + (
                self.knots[i + k] - xs
            ) * self._basis(xs, i + 1, k - 1)
            return (v * k) / ((k - 1) * (self.knots[i + k] - self.knots[i]))

    def _bases(self, xs):
        return [self._basis(xs, i, k=self.order) for i in range(self.N)]

    def bases(self, xs):
        return np.concatenate(self._bases(xs)).reshape(self.N, *xs.shape)

    def project(self, bases, shape, coefs):
        coefs = coefs.reshape(shape)
        coefs /= np.sum(coefs)
        return np.sum(coefs * bases, axis=0) * self.norm(coefs)

    def eval(self, xs, shape, coefs):
        return self.project(self.bases(xs), shape, coefs)

    def __call__(self, xs, coefs):
        return self.eval(xs, (-1, 1), coefs)

order_map = {0: "Constant", 1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic", 5: "Quintic"}

xmin = 0
xmax = 1


class PlotSplineBasis(Scene):
    def setup_basis(self, k, ndof=25):
        nadd = k - 1
        interior_knots = np.linspace(xmin, xmax, ndof+k-2*nadd)
        dx = interior_knots[1]-interior_knots[0]
        knots = np.concatenate([xmin-np.arange(1,k)[::-1]*dx, interior_knots, xmax+dx*np.arange(1,k)])
        coef = np.array([1./ndof]*ndof)
        grid = np.linspace(xmin, xmax, 1000)
        basis = np.array(BasisSpline(ndof, k=k, knots=knots).bases(grid))
        color_cycle = iter(plt.cm.get_cmap('magma')(np.linspace(0.1, 1.0, ndof+1, endpoint=True)[::-1]))
        total_color = rgba_to_color(next(color_cycle))
        return coef, grid, basis, color_cycle, total_color
    
    def init_axes(self, k, ndof):
        ax = Axes(
            x_range=[0, 1], y_range=[0, 1.5, 0.1], axis_config={"include_tip": False}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="B_{k}(x)")
        self.add(ax, labels)
        title1 = Text(f"{order_map[k-1]} (k={k-1})", font_size=32).to_edge(UP).shift(2*LEFT)
        title2 = Text(f"Spline Basis: {ndof} knots", font_size=32).next_to(title1, RIGHT)
        eq = MathTex("B_{k}(x) = \\sum_{i}^{N_\\mathrm{dof}} c_{i} B_{i,k}(x)").next_to(title1, DOWN).shift(2*RIGHT)
        return ax, title1, title2, eq

    def update_text(self, k):
        title1 = Text(f"{order_map[k-1]} (k={k-1})", font_size=32).to_edge(UP).shift(2*LEFT)
        return title1
    
    def draw_basis_component(self, ax, xs, ys, color):
        line = ax.animate.plot_line_graph(xs, ys, line_color=color)
        return line
    
    def plot_spline_curves(self, ax, ndof, coef, grid, basis, color_cycle, total_color):
        total = 0
        curves = []
        for ii in range(ndof):
            basis[ii,0] = 2*basis[ii,1]-basis[ii,2]
            basis[ii,-1] = 2*basis[ii,-2]-basis[ii,-3]
            norm = 1.0 / (sum([np.trapz(basis[i, :], grid) * coef[i] for i in range(ndof)]))
            c=rgba_to_color(next(color_cycle))
            curve = ax.plot_line_graph(grid[np.nonzero(basis[ii,:])], basis[ii,:][np.nonzero(basis[ii,:])] * coef[ii] * norm, line_color=c, add_vertex_dots=False, stroke_width=8)
            curves.append(curve)
            total += norm * coef[ii] * basis[ii,:]
        total_line = ax.plot_line_graph(grid, total, line_color=total_color, add_vertex_dots=False, stroke_width=10)
        curves.append(total_line)
        return curves
    
    def construct(self):
        N = 30
        eq = None
        ks = range(6)
        ax,tit1,tit2,eq = self.init_axes(ks[0]+1, N)
        for i, order in enumerate(ks):
            coef, grid, basis, color_cycle, total_color = self.setup_basis(order+1, ndof=N)
            if i == 0:
                self.play(Write(tit1))
                self.play(Write(tit2))
                self.play(Write(eq))                
            else:
                self.remove(tit1)                
                tit1 = self.update_text(order+1)
                self.play(Write(tit1))
            lines = self.plot_spline_curves(ax, N, coef, grid, basis, color_cycle, total_color)
            plot_anims = [Create(c) for c in lines]
            self.play(AnimationGroup(*plot_anims, lag_ratio=0.1))
            self.wait(duration=1.35)
            animations = [FadeOut(c) for c in lines]
            self.play(AnimationGroup(*animations, lag_ratio=0.05))