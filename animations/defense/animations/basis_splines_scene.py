from manim import *
import numpy as np
from scipy.interpolate import BSpline
import pickle
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


class BasisSplines(Scene):
    def construct_basis_components(self, k, ts, x):
        N = len(ts)-k-1
        return BSpline(ts, np.diag(np.ones(N)), k)(x).T
    
    def create_colorset(self, N):
        color_cycle = iter(plt.cm.get_cmap('magma')(np.linspace(0.1, 0.9, N+1, endpoint=True)[::-1]))
        total_color = rgba_to_color(next(color_cycle))
        basis_colors = [rgba_to_color(next(color_cycle)) for i in range(N)]
        return basis_colors, total_color

    def construct(self):
        with open("../basis_spline_data_inference.pkl", "rb") as f:
            sample_data_dict = pickle.load(f)
        X = sample_data_dict['X']
        Y = sample_data_dict['Y']
        X_test = sample_data_dict['X_test']
        posterior_samples = sample_data_dict['samples']
        knots = sample_data_dict['knots']
        curves = sample_data_dict['curves']
        dmat_test = sample_data_dict['dmat_test']
        Nknots = len(knots)-4
        basis_colors, total_color = self.create_colorset(Nknots)
        N = len(curves)
        low,high = knots[0], knots[-1]
        grid = NumberPlane(x_range=[low+0.55,high-0.55], y_range=[-2.4,2.6], y_length=4.5, x_length=12, faded_line_ratio=2).shift(0.75*UP)
        gaxx = grid.get_x_axis()
        gaxy = grid.get_y_axis()
        gaxx.add_numbers(x_values=np.linspace(-1,1,3))
        gaxy.add_numbers(x_values=np.linspace(-2,2,5))
        self.add(grid)
        title = Text(f"Basis Splines: {Nknots}-knots", font_size=32).to_edge(UP)
        self.play(Write(title))

        for k in [3]:
            msg = Text(f"Degree={k}", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
            self.play(Write(msg))
            dmat = self.construct_basis_components(k, knots, X_test)
            total = np.sum(dmat, axis=0)*2
            coefs = np.ones(Nknots)*2
            knots_coords = [[knots[i+2],coefs[i],0] for i in range(len(coefs))]
            knots_dots = [Dot(grid.c2p(*point), radius=1.25*DEFAULT_DOT_RADIUS, fill_opacity=1, color=basis_colors[i]) for i,point in enumerate(knots_coords)]
            self.add_foreground_mobjects(*knots_dots)
            self.play(AnimationGroup(*[Create(d) for d in knots_dots], lag_ratio=0.15))
            basis_lines = [grid.plot_line_graph(X_test, dmat[i,:]*coefs[i], line_color=basis_colors[i], add_vertex_dots=False, stroke_width=4.) for i in range(Nknots)]
            total_line = grid.plot_line_graph(X_test, total, line_color=total_color, add_vertex_dots=False, stroke_width=4.)
            self.play(AnimationGroup(*[Create(l) for l in [*basis_lines, total_line]], lag_ratio=0.15))
            self.play(FadeOut(msg))
            self.wait(0.5)
            self.play(*[FadeOut(mo) for mo in [*knots_dots, *basis_lines, total_line]])
            self.wait(0.1)

        self.play(FadeOut(title))


        new_title = Text(f"Basis Spline Regression", font_size=32).to_edge(UP)
        self.play(Write(new_title))

        resid_grid = NumberPlane(x_range=[low+0.55,high-0.55], y_range=[-1,1], y_length=1.5, x_length=9, ).next_to(grid, DOWN, buff=0.3).shift(2.5*LEFT)
        raxx = resid_grid.get_x_axis()
        raxy = resid_grid.get_y_axis()
        raxx.add_numbers(x_values=np.linspace(-1,1,3))
        raxy.add_numbers(x_values=np.linspace(-1,1,3))
        resid_dens_grid = NumberPlane(x_range=[-1.0,1.0], y_range=[0,1.0], y_length=1.5, x_length=4, ).next_to(grid, DOWN, buff=0.3).shift(4.5*RIGHT)
        self.add(resid_grid, resid_dens_grid)
        rtit1=Text("Residuals", font_size=16).next_to(resid_grid, UP, buff=0.125)
        self.play(Write(rtit1))
        rtit2 = Text("Residual Histogram", font_size=16).next_to(resid_dens_grid, UP, buff=0.125)
        self.play(Write(rtit2))
        msg = Text("Add Data", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
        self.play(Write(msg))
        datapoints_coords = [[X[i],Y[i],0] for i in range(len(X))]
        datapoints = [Dot(grid.c2p(*point), radius=0.5*DEFAULT_DOT_RADIUS, fill_opacity=0.9, color=WHITE) for point in datapoints_coords]
        self.play(AnimationGroup(*[Create(d) for d in datapoints], lag_ratio=0.01))
        self.play(FadeOut(msg))
        msg = Text("Construc Basis Set", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
        self.play(Write(msg))

        # Place Basis Components
        coefs = np.ones(Nknots)*2
        knots_coords = [[knots[i+2],coefs[i],0] for i in range(len(coefs))]
        knots_dots = [Dot(grid.c2p(*point), radius=1.25*DEFAULT_DOT_RADIUS, fill_opacity=1, color=basis_colors[i]) for i,point in enumerate(knots_coords)]
        self.add_foreground_mobjects(*knots_dots)
        total = np.einsum("i...,i->...", dmat_test, coefs)
        basis_lines = [grid.plot_line_graph(X_test, dmat_test[i]*coefs[i], line_color=basis_colors[i], add_vertex_dots=False, stroke_width=4.) for i in range(Nknots)]
        total_line = grid.plot_line_graph(X_test, total, line_color=total_color, add_vertex_dots=False, stroke_width=4.)
        self.play(AnimationGroup(*[Create(d) for d in knots_dots], lag_ratio=0.15))
        self.play(AnimationGroup(*[Create(l) for l in [*basis_lines, total_line]], lag_ratio=0.15))
        self.wait(0.25)

        self.play(FadeOut(msg))
        msg = Text("Infer Weights", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
        self.play(Write(msg))
        self.wait(0.05)
        self.play(FadeOut(msg))
        dmat_X = self.construct_basis_components(3, knots, X)
        thin_lines = []
        rs = []
        for i in range(10):
            idx = np.random.randint(0,N)
            cs = posterior_samples['cs'][idx,:]
            sigma = posterior_samples['sigma'][idx]
            knots_coords = [[knots[j+2],cs[j],0] for j in range(Nknots)]
            new_knots_dots = [Dot(grid.c2p(*point), radius=1.25*DEFAULT_DOT_RADIUS, fill_opacity=1, color=basis_colors[i]) for i,point in enumerate(knots_coords)]
            resids = np.einsum("i...,i->...", dmat_X, cs) - Y
            new_total_line = grid.plot_line_graph(X_test, curves[idx,:], line_color=total_color, add_vertex_dots=False, stroke_width=4.)
            thin_line = grid.plot_line_graph(X_test, curves[idx,:], line_color=total_color, add_vertex_dots=False, stroke_width=1.)
            thin_lines.append(thin_line)
            new_basis_lines = [grid.plot_line_graph(X_test, dmat_test[i]*cs[i], line_color=basis_colors[i], add_vertex_dots=False, stroke_width=4.) for i in range(Nknots)]
            resid_coords = [[X[j],resids[j],0] for j in range(len(X))]
            resid_dots_big = [Dot(resid_grid.c2p(*point), radius=0.5*DEFAULT_DOT_RADIUS, fill_opacity=1, color=WHITE) for point in resid_coords]
            resid_dots = [Dot(resid_grid.c2p(*point), radius=0.1*DEFAULT_DOT_RADIUS, fill_opacity=1, color=WHITE) for point in resid_coords]
            resid_xgrid = np.linspace(-1,1,100)
            kde_curve = gaussian_kde(resids)(resid_xgrid)*0.8
            thick_kde_line = resid_dens_grid.plot_line_graph(resid_xgrid, kde_curve, line_color=total_color, add_vertex_dots=False, stroke_width=4.)
            kde_line = resid_dens_grid.plot_line_graph(resid_xgrid, kde_curve, line_color=total_color, add_vertex_dots=False, stroke_width=1.)
            self.play(*[*[Transform(knots_dots[j], new_knots_dots[j]) for j in range(len(knots_dots))], *[Transform(basis_lines[j], new_basis_lines[j]) for j in range(len(basis_lines))], Transform(total_line, new_total_line)])
            self.add(thin_line)
            self.play(*[thick_kde_line.animate(), *[Create(d) for d in resid_dots_big]], lag_ratio=0.0)
            self.play(*[Transform(thick_kde_line, kde_line), *[Transform(resid_dots_big[j], resid_dots[j]) for j in range(len(resid_dots_big))]])
            thin_lines.append(thick_kde_line)
            rs.append(resid_dots_big)
            self.wait(0.1)

        self.wait(6)
        mos = [*resid_dots, *basis_lines, *thin_lines, total_line, kde_line, thick_kde_line, *resid_dots_big, *knots_dots, *datapoints, grid, resid_grid, resid_dens_grid, new_title, rtit1, rtit2]
        for r in rs:
            self.remove(*r)
        self.remove(*mos)
        self.wait(3)

class SmoothingPrior(Scene):
    def construct(self):
        pass

class SmoothingPriorData(Scene):
    def construct(self):
        pass