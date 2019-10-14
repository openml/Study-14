import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# The following two classes allow circumventing OpenML's "only one of a flow as
# a component' restriction (it lets us use the same part of the pipeline twice).

class CategoricalImputer(SimpleImputer):
    """Subclass of ``sklearn.impute.SimpleImputer`` to circumvent OpenML's limitation of one
    flow being only once per run (i.e. it cannot appear twice as a subflow)."""

    def __init__(self):
        # By definition of the benchmark suite, only
        # 5000 categories can occur per dataset. Using
        # this value to impute missing values prior to
        # OHE is a valid strategy as encoding strings/
        # categories can only have a maximal value of
        # 5000
        self.missing_values = np.nan
        self.strategy = "constant"
        self.fill_value = 999999999
        self.verbose = 0
        self.copy = True
        self.add_indicator = False


class NumericalImputer(SimpleImputer):
    """Subclass of ``sklearn.impute.SimpleImputer`` to circumvent OpenML's limitation of one
    flow being only once per run (i.e. it cannot appear twice as a subflow)."""

    def __init__(self):
        self.missing_values = np.nan
        self.strategy = 'median'
        self.fill_value = None
        self.verbose = 0
        self.copy = True
        self.add_indicator = True


# The following two classes allow circumventing OpenML's "only one of a flow as
# a component' restriction (it lets us use the same part of the pipeline twice).
class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, self.indices]


class NumericalFeatureSelector(FeatureSelector):
    """Subclass of ``openmlstudy99.preprocessing.FeatureSelector`` to circumvent OpenML's
    limitation of one flow being only once per run (i.e. it cannot appear twice as a subflow)."""
    pass


class CategoricalFeatureSelector(FeatureSelector):
    """Subclass of ``openmlstudy99.preprocessing.FeatureSelector`` to circumvent OpenML's
    limitation of one flow being only once per run (i.e. it cannot appear twice as a subflow)."""
    pass
