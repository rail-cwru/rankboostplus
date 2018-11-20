import sys
import numpy as np
from rankboost.base_ranking import RankBoost


class RBC(RankBoost):
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
            rt = rts[choice]
            if rt in (-1, 0, 1):
                if rt == -1:
                    alpha = -sys.maxsize
                elif rt == 1:
                    alpha = sys.maxsize
                else:
                    break
                self.alphas.append(alpha)
                self.classifiers.append(feature_thresholds[choice])
                break

            alpha = np.log((1 + rt) / (1 - rt)) / 2.0
            self.alphas.append(alpha)
            self.classifiers.append(feature_thresholds[choice])
            w *= np.exp(-alpha * rankers[:, choice])
            w /= w.sum()
