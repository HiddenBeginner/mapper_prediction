import numpy as np
from inspect import signature
from joblib import Parallel, delayed

from sklearn.cluster import DBSCAN
from sklearn.base import clone
from sklearn.utils import check_array
from gtda.mapper.cluster import ParallelClustering


def _sample_weight_computer(rel_indices, sample_weight):
    return {"sample_weight": sample_weight[rel_indices]}


def _empty_dict(*args):
    return {}


def _indices_computer_precomputed(rel_indices):
    return np.ix_(rel_indices, rel_indices)


def _indices_computer_not_precomputed(rel_indices):
    return rel_indices


class PredictiveDBSCAN(DBSCAN):
    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        """
        Reference:
            https://stackoverflow.com/a/51516334
        """
        n_samples = len(X)
        y = -1 * np.ones(n_samples, dtype=int)
        for i in range(n_samples):
            if len(self.components_) > 0:
                diff = self.components_ - X[i, :]
                dist = np.linalg.norm(diff, axis=1)
                shortest_dist_idx = np.argmin(dist)
                if dist[shortest_dist_idx] < self.eps:
                    y[i] = self.labels_[self.core_sample_indices_[shortest_dist_idx]]
        return y


class PredictiveParallelClustering(ParallelClustering):
    def fit(self, X, y=None, sample_weight=None):
        """Fit the clusterer on each portion of the data.

        :attr:`clusterers_` and :attr:`clusters_` are computed and stored.

        Parameters
        ----------
        X : list-like of form ``[X_tot, masks]``
            Input data as a list of length 2. ``X_tot`` is an ndarray of shape
            (n_samples, n_features) or (n_samples, n_samples) specifying the
            full data. ``masks`` is a boolean ndarray of shape
            (n_samples, n_portions) whose columns are boolean masks
            on ``X_tot``, specifying the portions of ``X_tot`` to be
            independently clustered.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        sample_weight : array-like or None, optional, default: ``None``
            The weights for each observation in the full data. If ``None``,
            all observations are assigned equal weight. Otherwise, it has
            shape (n_samples,).

        Returns
        -------
        self : object

        """
        X_tot, masks = X
        check_array(X_tot, ensure_2d=True)
        check_array(masks, ensure_2d=True)
        if not np.issubdtype(masks.dtype, bool):
            raise TypeError("`masks` must be a boolean array.")
        if len(X_tot) != len(masks):
            raise ValueError("`X_tot` and `masks` must have the same number "
                             "of rows.")
        self._validate_clusterer()

        fit_params = signature(self.clusterer.fit).parameters
        if sample_weight is not None and "sample_weight" in fit_params:
            self._sample_weight_computer = _sample_weight_computer
        else:
            self._sample_weight_computer = _empty_dict

        if self._precomputed:
            self._indices_computer = _indices_computer_precomputed
        else:
            self._indices_computer = _indices_computer_not_precomputed

        # This seems necessary to avoid large overheads when running fit a
        # second time. Probably due to refcounts. NOTE: Only works if done
        # before assigning labels_single. TODO: Investigate
        self.clusterers = [clone(self.clusterer) for _ in range(len(masks.T))]
        self.labels_ = None

        labels_single = Parallel(n_jobs=self.n_jobs,
                                 prefer=self.parallel_backend_prefer)(
            delayed(self._labels_single)(
                X_tot[self._indices_computer(rel_indices)],
                rel_indices,
                sample_weight,
                clusterer_index
                )
            for clusterer_index, rel_indices in enumerate(map(np.flatnonzero, masks.T))
            )

        self.labels_ = np.empty(len(X_tot), dtype=object)
        self.labels_[:] = [tuple([])] * len(X_tot)
        for i, (rel_indices, partial_labels) in enumerate(labels_single):
            n_labels = len(partial_labels)
            labels_i = np.empty(n_labels, dtype=object)
            labels_i[:] = [((i, partial_label),)
                           for partial_label in partial_labels]
            self.labels_[rel_indices] += labels_i

        return self

    def _labels_single(self, X, rel_indices, sample_weight, clusterer_index):
        if len(X) == 0:
            return rel_indices, []
        clusterer = self.clusterers[clusterer_index]
        kwargs = self._sample_weight_computer(rel_indices, sample_weight)

        return rel_indices, clusterer.fit(X, **kwargs).labels_

    def _predict_single(self, X, rel_indices, sample_weight, clusterer_index):
        clusterer = self.clusterers[clusterer_index]
        if not hasattr(clusterer, 'components_'):
            return rel_indices, np.array([-1] * len(X))

        return rel_indices, clusterer.predict(X)

    def predict(self, X, y=None, sample_weight=None):
        X_tot, masks = X

        labels_single = Parallel(n_jobs=self.n_jobs,
                                 prefer=self.parallel_backend_prefer)(
                delayed(self._predict_single)(
                    X_tot[self._indices_computer(rel_indices)],
                    rel_indices,
                    sample_weight,
                    clusterer_index
                    )
                for clusterer_index, rel_indices in enumerate(map(np.flatnonzero, masks.T))
                )
        labels_ = np.empty(len(X_tot), dtype=object)
        labels_[:] = [tuple([])] * len(X_tot)
        for i, (rel_indices, partial_labels) in enumerate(labels_single):
            n_labels = len(partial_labels)
            labels_i = np.empty(n_labels, dtype=object)
            labels_i[:] = [((i, partial_label),)
                           for partial_label in partial_labels]
            labels_[rel_indices] += labels_i

        return labels_

    def transform(self, X, y=None, sample_weight=None):
        return self.predict(X, y, sample_weight)

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X, y, sample_weight)
