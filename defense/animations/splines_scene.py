from manim import *
import numpy as np
from scipy.interpolate import interp1d
import pickle
from scipy.stats import gaussian_kde

class PolynomialSplines(Scene):
    def construct_splines(self, grid, xpts, ypts, xrange=(0,1), kind='cubic'):
        points = [[x,y,0] for x,y in zip(xpts, ypts)]
        dots = [Dot(grid.c2p(*point), radius=2*DEFAULT_DOT_RADIUS) for point in points]
        interpolator = interp1d(xpts, ypts, kind=kind)
        splfct = lambda t: grid.c2p(t, interpolator(t), 0)
        spl = ParametricFunction(
                splfct,
                t_range=[*xrange],
                color=RED, stroke_width=5
            )
        return dots, spl
    
    def create_knots_and_splines(self, grid, xpts, ypts, kind='cubic', xrange=(0,1)):
        dots, spline = self.construct_splines(grid, xpts, ypts, kind=kind, xrange=xrange)
        self.play(*[Create(d) for d in dots])
        self.wait(0.2)
        self.play(Create(spline))
        self.wait(1)
        return dots, spline
    
    def transform_spline(self, grid, xpts, ypts, olddots, oldspl, kind='cubic', xrange=(0,1)):
        dots, spl = self.construct_splines(grid, xpts, ypts, kind=kind, xrange=xrange)
        anims = [Transform(olddots[i], dots[i]) for i in range(len(dots))]
        anims.append(Transform(oldspl, spl))
        self.play(*anims)
        return olddots, oldspl
    
    def animate_var_height_polynomial_spline(self, N, grid, kind, create=False, xrange=(0,1), dots=None, spl=None):
        if create:
            xpts = np.linspace(*xrange,N)
            ypts = np.zeros_like(xpts)
            dots, spl = self.create_knots_and_splines(grid, xpts, ypts, kind=kind, xrange=xrange)
        else:
            xpts = np.linspace(*xrange,N)
            ypts = np.zeros_like(xpts)
            dots, spl = self.transform_spline(grid, xpts, ypts, dots, spl, kind=kind, xrange=xrange)
            self.wait(0.2)

        
        new_ypts = np.array([1., -1., 1., -1., 1., -1., 1., -1., 1., -1.])
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.2)

        new_ypts *= -1.
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.2)

        new_ypts *= -1.5
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.2)

        new_ypts *= -2./3.
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.2)

        self.wait(.2)
        self.wiggle_knot_y(2, 1.25, grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(.05)
        self.wiggle_knot_y(6, 1.25, grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(.2)
        return dots, spl
    
    def wiggle_knot_y(self, idx, length, grid, xpts, ypts, dots, spl, kind, xrange=(0,1)):
        new_ypts = ypts.copy()
        new_ypts[idx] += length/2.
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        new_ypts[idx] -= length
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        new_ypts[idx] += length/2.
        dots, spl = self.transform_spline(grid, xpts, new_ypts, dots, spl, kind=kind, xrange=xrange)
        return dots, spl
    
    def wiggle_knot_x(self, idx, length, grid, xpts, ypts, dots, spl, kind, xrange=(0,1)):
        new_xpts = xpts.copy()
        new_xpts[idx] += length/2.
        dots, spl = self.transform_spline(grid, new_xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        new_xpts[idx] -= length/2.
        dots, spl = self.transform_spline(grid, new_xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        new_xpts[idx] -= length/2.
        dots, spl = self.transform_spline(grid, new_xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        new_xpts[idx] += length/2.
        dots, spl = self.transform_spline(grid, new_xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        return dots, spl
    
    def animate_var_loc_polynomial_spline(self, N, grid, kind, dots, spl, xrange=(0,1)):
        ypts = np.array([1., -1., 1., -1., 1., -1., 1., -1., 1., -1.])*-1.0
        xpts = np.linspace(*xrange,N)
        dots, spl = self.transform_spline(grid, xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.05)
        self.wiggle_knot_x(2, 0.6, grid, xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.05)
        self.wiggle_knot_x(6, 0.6, grid, xpts, ypts, dots, spl, kind=kind, xrange=xrange)
        self.wait(0.05)
        return dots, spl

    def construct(self):
        _scale_X = 2
        with open("../spline_data_inference.pkl", "rb") as f:
            sample_data_dict = pickle.load(f)
        X = sample_data_dict['X']*_scale_X
        Y = sample_data_dict['Y']
        X_test = sample_data_dict['X_test']*_scale_X
        posterior_samples = sample_data_dict['samples']
        knots = sample_data_dict['knots']*_scale_X
        curves = sample_data_dict['curves']
        Nknots = len(knots)
        N = len(curves)
        low,high = np.min(X), np.max(X)
        grid = NumberPlane(x_range=[low-0.05,high+0.05], y_range=[-3,3], y_length=4.5, x_length=12, faded_line_ratio=2).shift(0.75*UP)
        gaxx = grid.get_x_axis()
        gaxy = grid.get_y_axis()
        gaxx.add_numbers(x_values=np.linspace(low,high,5))
        gaxy.add_numbers(x_values=np.linspace(-2,2,5))
        self.add(grid)
        title = Text(f"Cubic Spline Interpolation: {Nknots}-knots", font_size=32).to_edge(UP)
        self.play(Write(title))

        dots, spl = self.animate_var_height_polynomial_spline(Nknots, grid, kind='cubic', create=True, xrange=(low,high))
        self.wait(.2)

        dots, spl = self.animate_var_loc_polynomial_spline(Nknots, grid, kind='cubic', dots=dots, spl=spl, xrange=(low,high))
        self.wait(0.8)
        self.play(FadeOut(title))
        self.wait(0.8)
        self.play(FadeOut(*[spl, *dots]))
        new_title = Text(f"Spline Regression", font_size=32).to_edge(UP)
        self.play(Write(new_title))

        resid_grid = NumberPlane(x_range=[low-0.05,high+0.05], y_range=[-1,1], y_length=1.5, x_length=9, ).next_to(grid, DOWN, buff=0.25).shift(2.5*LEFT)
        raxx = resid_grid.get_x_axis()
        raxy = resid_grid.get_y_axis()
        raxx.add_numbers(x_values=np.linspace(low,high,5))
        raxy.add_numbers(x_values=np.linspace(-1,1,3))
        resid_dens_grid = NumberPlane(x_range=[-1.0,1.0], y_range=[0,1.0], y_length=1.5, x_length=4, ).next_to(grid, DOWN, buff=0.25).shift(4.5*RIGHT)
        self.add(resid_grid, resid_dens_grid)
        rtit1=Text("Residuals", font_size=16).next_to(resid_grid, UP, buff=0.125)
        self.play(Write(rtit1))
        rtit2 = Text("Residual Histogram", font_size=16).next_to(resid_dens_grid, UP, buff=0.125)
        self.play(Write(rtit2))
        msg = Text("Add Data", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
        self.play(Write(msg))
        datapoints_coords = [[X[i],Y[i],0] for i in range(len(X))]
        datapoints = [Dot(grid.c2p(*point), radius=0.6*DEFAULT_DOT_RADIUS, fill_opacity=1, color=WHITE) for point in datapoints_coords]
        self.play(AnimationGroup(*[Create(d) for d in datapoints], lag_ratio=0.01))
        self.play(FadeOut(msg))
        msg = Text("Space Out Knots", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
        self.play(Write(msg))
        knots_coords = [[knots[i],0,0] for i in range(len(knots))]
        knots_dots = [Dot(grid.c2p(*point), radius=1.5*DEFAULT_DOT_RADIUS, fill_opacity=1, color=RED) for point in knots_coords]
        self.add_foreground_mobjects(*knots_dots)
        self.play(AnimationGroup(*[Create(d) for d in knots_dots], lag_ratio=0.05))
        self.play(FadeOut(msg))
        msg = Text("Infer Knot Heights", font_size=24).move_to(grid.c2p((low+high)/2.,2.5,0))
        self.play(Write(msg))
        self.wait(0.05)
        self.play(FadeOut(msg))
        sel = (X_test >= knots[0]) & (X_test <= knots[-1])
        lines = []
        rs = []
        for i in range(7):
            idx = np.random.randint(0,N)
            hts = posterior_samples['spl_hts'][idx,:]
            sigma = posterior_samples['sigma'][idx]
            knots_coords = [[knots[j],hts[j],0] for j in range(len(knots))]
            new_knots_dots = [Dot(grid.c2p(*point), radius=1.5*DEFAULT_DOT_RADIUS, fill_opacity=1, color=RED) for point in knots_coords]
            spl = interp1d(knots, hts, kind='cubic')
            resids = spl(X) - Y
            thick_line = grid.plot_line_graph(X_test[sel], curves[idx,sel], line_color=RED, add_vertex_dots=False, stroke_width=6.) #['line'].set_opacity(0.5)
            line = grid.plot_line_graph(X_test[sel], curves[idx,sel], line_color=RED, add_vertex_dots=False, stroke_width=1.) #['line'].set_opacity(0.5)
            resid_coords = [[X[i],resids[i],0] for i in range(len(X))]
            resid_dots_big = [Dot(resid_grid.c2p(*point), radius=0.75*DEFAULT_DOT_RADIUS, fill_opacity=1, color=WHITE) for point in resid_coords]
            resid_dots = [Dot(resid_grid.c2p(*point), radius=0.15*DEFAULT_DOT_RADIUS, fill_opacity=1, color=WHITE) for point in resid_coords]
            resid_xgrid = np.linspace(-1,1,100)
            kde_curve = gaussian_kde(resids)(resid_xgrid)*0.8
            thick_kde_line = resid_dens_grid.plot_line_graph(resid_xgrid, kde_curve, line_color=RED, add_vertex_dots=False, stroke_width=6.)
            kde_line = resid_dens_grid.plot_line_graph(resid_xgrid, kde_curve, line_color=RED, add_vertex_dots=False, stroke_width=1.)
            self.play(*[Transform(knots_dots[j], new_knots_dots[j]) for j in range(len(knots_dots))])
            self.play(AnimationGroup(*[thick_line.animate(), thick_kde_line.animate(), *[Create(d) for d in resid_dots_big]]), lag_ratio=0.0)
            self.play(*[Transform(thick_line, line), Transform(thick_kde_line, kde_line), *[Transform(resid_dots_big[i], resid_dots[i]) for i in range(len(resid_dots_big))]])
            lines.append(thick_line)
            lines.append(thick_kde_line)
            rs.append(resid_dots_big)
            self.wait(0.05)

        self.wait(6)
        mos = [*resid_dots, thick_line, kde_line, thick_kde_line, *resid_dots_big, *knots_dots, line, *datapoints, *lines, grid, resid_grid, resid_dens_grid, new_title, rtit1, rtit2]
        for r in rs:
            self.remove(*r)
        self.remove(*mos)
        self.wait(3)