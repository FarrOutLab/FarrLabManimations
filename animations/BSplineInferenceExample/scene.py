from manim import *
import numpy as np
import deepdish as dd


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


class BSplineRegressionExample(Scene):
    def construct(self):
        self.nknot_list = [6,8,10,12,16,20,30,40]
        self.data_files = [f'media/data/BasisSplineExample_{i}knots_60train_0.2sigma_inference_data.h5' for i in self.nknot_list]
        self.posteriors = {}
        self.data = None
        for data_file,n in zip(self.data_files,[6,8,10,12,16,20,30,40]):
            data = dd.io.load(data_file)
            self.posteriors[n] = data['samples']
            if self.data is None:
                self.data = data['data']
        self.show_data()
        self.wait(3)
        
        for n in self.nknot_list:
            self.show_posterior(n)
            self.wait(1)
            self.clear_posterior()
        
    def show_data(self):
        return
    
    def animate_spline_curve(self,dmat,cs):
        return
    
    def show_posterior(self,n,N_plot=50):
        dmat = BasisSpline(n, xrange=(-1.0,1.0)).bases(self.data['X_test']).T
        self.animate_spline_curve(dmat,cs=np.ones(dmat.shape[1]))
        for _ in range(N_plot):
            i = np.random.choice(self.posteriors[n]["cs"].shape[0],1)[0]
            cs = self.posteriors[n]["cs"][i]
            self.animate_spline_curve(dmat,cs)
        return
    
    def clear_posteror(self):
        return