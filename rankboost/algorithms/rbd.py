import sys

import numpy as np
from rankboost.base_ranking import RankBoost


class RBD(RankBoost):
    def fit(self, X, y):
        if type(X) != np.ndarray:
            X, y = self.normalize_dict(X, y)

        feature_thresholds, rankers = self.create_rankers(X, y)
        w = (np.ones(len(y)) / float(len(y))).astype(np.float32)

        self.alphas = []
        self.classifiers = []

        for iteration in range(self.niter):
            rts = w.dot(rankers)
            choice = np.argmax(np.abs(rts))
            epsilon_pos = w.dot(np.greater(rankers[:, choice], 0).astype(np.float32))
            epsilon_neg = w.dot(np.less(rankers[:, choice], 0).astype(np.float32))

            to_break = abs(epsilon_neg) < 1e-10 or abs(epsilon_pos) < 1e-10
            if iteration > 0 and to_break:
                break
            if abs(epsilon_neg) < 1e-10:
                alpha = sys.maxsize // 2
            elif abs(epsilon_pos) < 1e-10:
                alpha = -sys.maxsize // 2
            else:
                alpha = 0.5 * np.log(epsilon_pos / epsilon_neg)

            self.alphas.append(alpha)
            self.classifiers.append(feature_thresholds[choice])
            if to_break:
                break
            w *= np.exp(-alpha * rankers[:, choice])
            w /= w.sum()
