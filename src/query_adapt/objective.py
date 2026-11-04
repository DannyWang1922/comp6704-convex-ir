"""Pairwise logistic objective with explicit regularization around identity."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass
class PairwiseObjective:
    features: np.ndarray
    regularization: float

    def __post_init__(self):
        self.features = np.asarray(self.features, dtype=np.float64)
        if self.features.ndim != 2 or min(self.features.shape) == 0:
            raise ValueError("features must be a nonempty N x d matrix")
        if not np.isfinite(self.features).all():
            raise ValueError("features must be finite")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")

    @property
    def dimension(self):
        return self.features.shape[1]

    def _parameter(self, a):
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (self.dimension,) or not np.isfinite(a).all():
            raise ValueError("invalid parameter vector")
        return a

    def margins(self, a):
        return self.features @ self._parameter(a)

    def _validated_margins(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (len(self.features),) or not np.isfinite(z).all():
            raise ValueError("invalid cached margins")
        return z

    def value_from_margins(self, a, z):
        a = self._parameter(a)
        z = self._validated_margins(z)
        return float(np.logaddexp(0.0, -z).mean()
                     + 0.5 * self.regularization * np.dot(a - 1, a - 1))

    def gradient_from_margins(self, a, z):
        a = self._parameter(a)
        z = self._validated_margins(z)
        return -(self.features.T @ expit(-z)) / len(self.features) + self.regularization * (a - 1)

    def hessian_from_margins(self, z):
        z = self._validated_margins(z)
        x = self.features
        weights = expit(z) * expit(-z)
        return (x.T * weights) @ x / len(x) + self.regularization * np.eye(self.dimension)

    def value(self, a):
        a = self._parameter(a)
        z = self.features @ a
        return float(np.logaddexp(0.0, -z).mean()
                     + 0.5 * self.regularization * np.dot(a - 1, a - 1))

    def value_gradient(self, a):
        a = self._parameter(a)
        x = self.features
        z = x @ a
        value = np.logaddexp(0.0, -z).mean() + 0.5 * self.regularization * np.dot(a - 1, a - 1)
        gradient = -(x.T @ expit(-z)) / len(x) + self.regularization * (a - 1)
        return float(value), gradient

    def hessian(self, a):
        a = self._parameter(a)
        x = self.features
        z = x @ a
        weights = expit(z) * expit(-z)
        return (x.T * weights) @ x / len(x) + self.regularization * np.eye(self.dimension)
