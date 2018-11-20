import sys
import warnings
import time

import numpy as np
from scipy import linalg

from rankboost.base_ranking import RankBoost, get_duplicates_mask


class RBPFast(RankBoost):
    r"""
    {\sc Rankboost+} as described in the paper.
    """

    def fit(self, X, y):
        if type(X) != np.ndarray:
            X, y = self.normalize_dict(X, y)

        feature_thresholds, new_rankers = self.create_rankers(X, y)

        w = np.ones(len(y), dtype=np.float32) / float(len(y))

        good_ix = get_duplicates_mask(new_rankers)
        if len(good_ix) != len(feature_thresholds):
            feature_thresholds = [feature_thresholds[i] for i in good_ix]
            new_rankers = new_rankers[:, np.array(good_ix)]

        self.alphas = []
        self.classifiers = []
        self.zs = [1.0]
        self.timings = []
        self.timings.append(time.time())

        etas = np.zeros(0)
        accumulated = np.empty((len(w), 0), dtype=np.float32)
        accumulated_zero = np.empty((len(w), 0), dtype=np.float32)
        accumulated_classifiers = []
        greedy_regime = True

        for iteration in range(self.niter):
            independent = True
            if iteration > 0 and greedy_regime:
                new_scores = -w.dot(new_rankers)
                existing_scores = -w.dot(accumulated) + w.dot(
                    accumulated_zero
                ) * np.tanh(etas)
                curr_best = np.abs(existing_scores).argmax()

                new_best = np.abs(new_scores).argmax()

                if np.abs(existing_scores[curr_best]) >= np.abs(new_scores[new_best]):
                    chosen_ranker = accumulated[:, curr_best]
                    pre_alpha = etas[curr_best]
                    choice_ix = curr_best
                    independent = False
                    found = True
                else:
                    chosen_ranker = new_rankers[:, new_best].copy()
                    comb, resid, rank, _ = linalg.lstsq(accumulated, chosen_ranker)
                    found = not (resid.size == 0 or resid < 1e-5)

                    choice_ix = new_best
                    pre_alpha = 0.0

                    new_rankers[:, choice_ix] = 0.0
                    new_scores[choice_ix] = 0.0

                if not found:
                    greedy_regime = False
                    good_mask = np.abs(new_rankers).max(0) > 1e-8
                    if np.any(
                        good_mask
                    ):  # don't need to do anything if all rankers are zero
                        # First remove all of the zero columns
                        ix, = np.where(good_mask)
                        ix = np.random.permutation(ix)
                        clfs = [feature_thresholds[i] for i in ix]
                        new_rankers = new_rankers[:, ix]

                        # Remove all of the dependent columns from the concatenation
                        # We know we are going to keep the first accumulated.shape[1] of them
                        r, = linalg.qr(np.append(accumulated, new_rankers, 1), mode="r")
                        good_mask = (np.abs(np.diag(r)) > 1e-4)[accumulated.shape[1] :]

                        new_accumulated = new_rankers[:, np.where(good_mask)[0]].astype(
                            np.float32
                        )

                        new_classifiers = [clfs[i] for i in np.where(good_mask)[0]]
                        accumulated = np.concatenate(
                            (accumulated, new_accumulated), axis=1
                        ).astype(np.float32)
                        accumulated_zero = (accumulated == 0).astype(np.float32)
                        accumulated_classifiers.extend(new_classifiers)
                        etas = np.append(etas, np.zeros(len(new_classifiers))).astype(
                            np.float32
                        )

            elif iteration == 0:  # To avoid max(empty sequence) errors
                scores = w.dot(new_rankers)
                choice_ix = np.abs(scores).argmax()
                chosen_ranker = new_rankers[:, choice_ix].copy()
                new_rankers[:, choice_ix] = 0.0
                pre_alpha = 0.0

            if not greedy_regime and iteration > 0:
                scores = -w.dot(accumulated) + w.dot(accumulated_zero) * np.tanh(etas)

                choice_ix = np.abs(scores).argmax()
                chosen_ranker = accumulated[:, choice_ix]
                pre_alpha = etas[choice_ix]
                independent = False

            eps_zero = w.dot(np.equal(chosen_ranker, 0, dtype=np.float32))
            eps_neg = w.dot(np.less(chosen_ranker, 0, dtype=np.float32))
            eps_pos = w.dot(np.greater(chosen_ranker, 0, dtype=np.float32))

            num = eps_pos + eps_zero * np.exp(-pre_alpha) / (2 * np.cosh(pre_alpha))
            denom = eps_neg + eps_zero * np.exp(pre_alpha) / (2 * np.cosh(pre_alpha))

            to_break = num == 0 or denom == 0

            if num == 0:
                alpha = -sys.maxsize
            elif denom == 0:
                alpha = sys.maxsize
            else:
                alpha = np.log(num / denom) / 2.0

            self.alphas.append(alpha)

            if independent:
                self.classifiers.append(feature_thresholds[choice_ix])
            else:
                self.classifiers.append(accumulated_classifiers[choice_ix])

            if iteration > 2 and (self.classifiers[-1] == self.classifiers[-2]):
                warnings.warn("Chose two of the same rankers in a row; breaking")
                break

            if independent:
                accumulated_classifiers.append(feature_thresholds[choice_ix])
                accumulated = np.append(accumulated, chosen_ranker[:, None], axis=1)
                accumulated_zero = np.append(
                    accumulated_zero, chosen_ranker[:, None] == 0, axis=1
                )
                etas = np.append(etas, alpha)
            else:
                etas[choice_ix] += alpha

            if independent and to_break:
                break

            m = chosen_ranker == 0
            scalers = (
                np.exp(-alpha * chosen_ranker) * (~m)
                + np.cosh(alpha + pre_alpha) / np.cosh(pre_alpha) * m
            )
            w *= scalers
            w /= w.sum()
