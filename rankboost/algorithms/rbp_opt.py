import numpy as np
from rankboost.base_ranking import RankBoost
from scipy import linalg, optimize
import sys
import time


class RBPOpt(RankBoost):
    """
    Optimal {\sc Rankboost+} as briefly described in the paper
    """

    def fit(self, X, y):
        if type(X) != np.ndarray:
            X, y = self.normalize_dict(X, y)

        feature_thresholds, rankers = self.create_rankers(X, y)
        zero_rankers = rankers == 0

        w = np.ones(len(y), dtype=np.float32) / float(len(y))

        self.alphas = []
        self.classifiers = []
        self.zs = [1.0]
        self.timings = []
        self.timings.append(time.time())

        etas = np.zeros(0)
        accumulated = np.empty((len(w), 0))
        span_updated = True

        for iteration in range(self.niter):
            scores = -w.dot(rankers)
            if iteration > 0:
                if span_updated:
                    combs, resids, _, _ = linalg.lstsq(accumulated, rankers)

                if len(resids) == 0:
                    eqs = (np.abs(accumulated.dot(combs) - rankers) < 1e-5).all(0)
                else:
                    eqs = resids < 1e-6

                acc_epos = w.dot(accumulated > 0)
                acc_ez = w.dot(accumulated == 0)
                acc_eneg = 1 - acc_ez - acc_epos
                acc_scores = -acc_epos + acc_eneg + acc_ez * np.tanh(etas)
                scores[eqs] = acc_scores.dot(
                    combs[:, eqs] / np.linalg.norm(combs[:, eqs], axis=0)
                )

            choice = np.abs(scores).argmax()

            independent = (iteration == 0) or not eqs[choice]
            if not independent:
                comb = combs[:, choice]

            eps_zero = w.dot(rankers[:, choice] == 0)
            eps_neg = w.dot(rankers[:, choice] < 0)
            eps_pos = 1 - eps_neg - eps_zero
            pre_etas = etas.copy()

            if independent:
                num = eps_pos + eps_zero / 2.0
                denom = eps_neg + eps_zero / 2.0

                to_break = num == 0 or denom == 0

                if num == 0:
                    alpha = -sys.maxsize
                elif denom == 0:
                    alpha = sys.maxsize
                else:
                    alpha = np.log(num / denom) / 2.0
            else:
                alpha, _ = _alpha_search(
                    accumulated, comb, etas, w * self.zs[-1] * rankers.shape[0]
                )
                alpha = alpha[0]

            self.alphas.append(alpha)
            self.classifiers.append(feature_thresholds[choice])

            if independent:
                accumulated = np.append(accumulated, rankers[:, choice, None], axis=1)
                etas = np.append(etas, alpha)
                span_updated = True
            else:
                etas += alpha * comb
                span_updated = False

            if independent and to_break:
                break

            m = zero_rankers[:, choice]
            if independent:
                scalers = (
                    np.exp(-alpha * rankers[:, choice]) * (~m) + np.cosh(alpha) * m
                )
            else:
                scalers = (
                    np.exp(-alpha * rankers[:, choice]) * (~m)
                    + np.prod(np.cosh(etas) / np.cosh(pre_etas)) * m
                )
            w *= scalers
            self.zs.append(self.zs[-1] * w.sum())
            w /= w.sum()
            self.timings.append(time.time())

        self.accumulated = accumulated
        self.etas = etas


def _alpha_search(hs, betas, pre_alphas, unscaled_w):
    mz = np.empty_like(hs, dtype=np.float32)
    mn = np.empty_like(hs, dtype=np.float32)
    mp = np.empty_like(hs, dtype=np.float32)
    np.equal(hs, 0, mz)
    np.equal(hs, 1, mp)
    np.equal(hs, -1, mn)

    def f(x):
        return _objective(pre_alphas + betas * x, hs)

    def df(x):
        return np.array([_dobjective(unscaled_w, pre_alphas, x, betas, mz, hs)])

    res = optimize.minimize(f, 0.0, jac=df)
    return res.x, res.nit


def _objective(x, cols):
    m = (cols == 0) * 1
    scales = np.exp(-cols * x) * (1 - m) + m * np.cosh(x)
    out = np.exp(np.log(scales).sum(1)).mean()
    return out


def _dobjective(dists, alpha, x, betas, mz, hs):
    zz = dists.sum()
    dists = dists / zz
    zz /= len(dists)
    scb = x * betas
    nas = alpha + scb
    mid_part = -(hs - mz * np.tanh(nas)).dot(betas)
    updates = (
        np.exp(-hs * x * betas) * (1 - mz)
        + mz * np.cosh(alpha + x * betas) / np.cosh(alpha)
    ).prod(1)
    return dists.dot(mid_part * updates) * zz
