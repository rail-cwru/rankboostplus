from typing import List, Optional, Dict, Tuple, Any, TypeVar, Union, NamedTuple

import numpy as np

K = TypeVar("K")
XType = Union[np.ndarray, Dict[K, List[float]]]
YType = Union[np.ndarray, List[Tuple[K, K]]]


class StumpRanker(NamedTuple):
    feature: int
    threshold: float

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("Please input ndarray of rank 2")
        if y is not None and not isinstance(X, np.ndarray):
            raise ValueError("y must be an ndarray")
        preds: np.ndarray = (X[:, self.feature] > self.threshold) * 1
        if y is not None:
            preds = preds[y[:, 0]] - preds[y[:, 1]]
        return preds


class RankBoost:
    """
    Base class for all rankers.

    :param niter: upper bound on number of iterations to run the algorithm for
    """

    def __init__(self, niter: int = 10):
        self.niter = niter

        self.classifiers: Optional[List[StumpRanker]] = None
        self.alphas: Optional[List[float]] = None

    def signature(self) -> Dict[str, List]:
        """
        Get a signature that can be used to make predictions at every iteration
        """
        return {"alphas": self.alphas, "rankers": self.classifiers}

    @classmethod
    def from_signature(
        cls, alphas: List[float], rankers_signatures: List[Tuple[int, float]]
    ) -> "RankBoost":
        """
        Load a classifier from its signature

        :param alphas: Weights
        :type alphas: list[float]
        :param rankers_signatures: List of tuples representing StumpRankers
        :type rankers_signatures: List[Tuple[int, float]]
        :return: A RankBoost object
        """
        rb = cls()
        rb.alphas = alphas
        rb.classifiers = [StumpRanker(*i) for i in rankers_signatures]
        return rb

    @staticmethod
    def normalize_dict(
        X: Dict[K, List[float]], y: List[Tuple[K, K]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts data from dictionary and list to ndarrays.
        X is of the form {k1: [feature1, feature2, ...], k2: [feature1, feature2, ...], ...}
        y is a dictionary of tuples of keys of x, where y[ix][0] is ranked higher than y[ix][1]
        This will convert X into a ndarray xp, and y into an array that indexes X, so that
        np.array([X[p0] for p0, _ in y]) == X_new[y_new[:, 0]] and respectively with y[:, 1].

        :param X:
        :param y:
        :return:
        """
        index_to_key = {ix: k for ix, k in enumerate(sorted(X.keys()))}
        key_to_index = {k: ix for ix, k in index_to_key.items()}
        X_new = np.array([X[index_to_key[ix]] for ix in range(len(X))])
        y_new = []
        for p0, p1 in y:
            y_new.append((key_to_index[p0], key_to_index[p1]))
        return X_new, np.array(y_new)

    def fit(self, X: XType, y: YType) -> Any:
        """
        This should take either take ndarrays of the form that `normalize_dict`
        outputs, or dictionaries to be input to `normalize_dict`, and train the model
        """
        raise NotImplementedError("Please implement the fit function")

    def predict(self, X: XType, y: Optional[YType] = None):
        """
        Predict based on `self.classifiers` and `self.alphas`. Takes the same input types as `fit`.
        Outputs instance predictions if y is None, and otherwise outputs pairwise predictions.
        :param X:
        :param y:
        :return: np.ndarray of shape (len(X),) if y is None, or (len(y), ) if y is not None.
        """
        if len(self.alphas) == 0:
            return np.zeros(len(y))
        if type(X) != np.ndarray:
            X, y = self.normalize_dict(X, y)

        pred_dict = {c: c.predict(X) for c in set(self.classifiers)}
        preds = np.array([pred_dict[c] for c in self.classifiers]).T.dot(self.alphas)
        if y is not None:
            preds = preds[y[:, 0]] - preds[y[:, 1]]
        return preds

    def predict_cumulative(
        self, X: XType, y: Optional[YType] = None, max_iter: Optional[int] = None
    ):
        """
        Takes the same inputs as `predict`, but outputs cumulative predictions up to max_iter.

        :param X:
        :param y:
        :param max_iter: last iteration to make predictions for. Defaults to `len(self.classifiers)`.
        :return: np.ndarray of shape (len(X), max_iter) if y is None, or (len(y), max_iter) if y is not None.
        """
        if len(self.alphas) == 0:
            return np.zeros(len(y))
        if type(X) != np.ndarray:
            X, y = self.normalize_dict(X, y)
        if max_iter is None:
            max_iter = len(self.classifiers)
        pred_dict = {c: c.predict(X) for c in set(self.classifiers[:max_iter])}
        preds = np.array([pred_dict[c] for c in self.classifiers[:max_iter]]).T.astype(
            np.float32
        )
        preds *= np.array(self.alphas[:max_iter], dtype=np.float32)
        preds = np.cumsum(preds, axis=1)
        if y is not None:
            preds = preds[y[:, 0]] - preds[y[:, 1]]
        return preds

    @staticmethod
    def create_rankers(
        X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[StumpRanker], np.ndarray]:
        """
        Creates a set of Decision stump rankers. If a single feature has more than 255 levels,
        only 255 of the possible thresholds are chosen for rankers. Inputs must be of the form
        output by normalize_dict.

        :param X:
        :param y:
        :return:
        """
        feature_thresholds = []
        rankers = []
        for feature_num in range(X.shape[1]):  # build up classifiers/thresholds
            thresholds = np.convolve(np.unique(X[:, feature_num]), [0.5, 0.5], "valid")

            if len(thresholds) > 255:
                thresholds = np.random.choice(thresholds, 255, False)
            for t in thresholds:
                feature_thresholds.append(StumpRanker(feature_num, t))
                rc = (X[:, feature_num] > t).astype(np.float32)
                rc = rc[y[:, 0]] - rc[y[:, 1]]
                rankers.append(rc)
                # if len(np.unique(rc)) == 1:
                #     feature_thresholds.pop()
                #     rankers.pop()

        rankers = np.array(rankers, dtype=np.float32).T
        return feature_thresholds, rankers


class UnionFind:
    """
    Basic Union find class
    """

    def __init__(self, val=None):
        self.val = val
        self.parent = self
        self.rank = 1

    def find(self):
        # type: () -> UnionFind
        n = self
        while n is not n.parent:
            n = n.parent
        return n

    def union(self, y):
        # type: (UnionFind) -> None
        px, py = self.find(), y.find()
        if px is py:
            return
        py, px = sorted((px, py), key=lambda x: x.rank)
        py.parent = px
        px.rank += px.rank == py.rank


def get_duplicates_mask(rankers: np.ndarray) -> np.ndarray:
    """
    Retrieve a list of indices such that rankers[:, mask] contains
    no duplicate columns, and contains the same number of unique
    columns as before applying this function

    :param rankers: A MxN ndarray
    :type rankers: np.ndarray
    :rtype: np.ndarray[bool]
    """
    eqs = rankers.T.dot(rankers) == np.abs(rankers).sum(0)  # type: np.ndarray
    ufs = [UnionFind(i) for i in range(eqs.shape[0])]
    for i in range(len(ufs)):
        for j in range(i + 1, len(ufs)):
            if eqs[i, j]:
                UnionFind.union(ufs[i], ufs[j])
    del eqs
    good_ix = []
    seen_parents = set()
    for i in range(len(ufs)):
        p = ufs[i].find()
        if p in seen_parents:
            continue
        good_ix.append(i)
        seen_parents.add(p)
    return np.array(good_ix)
