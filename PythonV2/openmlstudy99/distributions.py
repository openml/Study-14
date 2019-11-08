import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
import scipy.stats


class OpenMLDistributionHelper:
    def _cdf(self, x, *args):
        raise NotImplementedError()

    def _sf(self, x, *args):
        raise NotImplementedError()

    def _ppf(self, q, *args):
        raise NotImplementedError()

    def _isf(self, q, *args):
        raise NotImplementedError()

    def _stats(self, *args, **kwds):
        raise NotImplementedError()

    def _munp(self, n, *args):
        raise NotImplementedError()

    def _entropy(self, *args):
        raise NotImplementedError()


class loguniform_gen(OpenMLDistributionHelper, rv_continuous):
    def _pdf(self, x, low, high):
        raise NotImplementedError()

    def _argcheck(self, low, high):
        self.a = low
        self.b = high
        return (high > low) and low > 0 and high > 0

    def logspace(self, num):
        start = np.log(self.a)
        stop = np.log(self.b)
        return np.logspace(start, stop, num=num, endpoint=True)

    def _rvs(self, low, high):
        low = np.log(low)
        high = np.log(high)
        return np.exp(self._random_state.uniform(low=low, high=high, size=self._size))
loguniform = loguniform_gen(name='loguniform')


class loguniform_int_gen(OpenMLDistributionHelper, rv_discrete):
    def _pmf(self, x, low, high):
        raise NotImplementedError()

    def _argcheck(self, low, high):
        self.a = low
        self.b = high
        return (high > low) and low >= 1 and high >= 1

    def _rvs(self, low, high):
        assert self.a >= 1
        low = np.log(low - 0.4999)
        high = np.log(high + 0.4999)
        return np.rint(
            np.exp(self._random_state.uniform(low=low, high=high, size=self._size))
        ).astype(int)
loguniform_int = loguniform_int_gen(name='loguniform_int')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Assumes that high is the target high + 1
    r = scipy.stats.randint(low=-10, high=11)
    samples = r.rvs(size=10000)
    assert np.max(samples) == 10, np.max(samples)
    assert np.min(samples) == -10

    r = scipy.stats.uniform(loc=0.1, scale=0.8)
    samples = r.rvs(size=1000000)
    assert 0.90001 >= np.max(samples) >= 0.8999, np.max(samples)
    assert 0.09999 <= np.min(samples) <= 0.1001, np.min(samples)

    r = loguniform(low=2**-12, high=2**12)
    samples = r.rvs(size=1000000)
    assert np.max(samples) <= 4096
    assert np.min(samples) >= 0.0000001

    r = loguniform(low=2 ** -12, high=2 ** 12)
    samples = r.rvs(size=1000000)
    assert np.max(samples) <= 4096
    assert np.min(samples) >= 0.0000001

    r = loguniform(low=1e-7, high=1e-1)
    samples = r.rvs(size=1000000)
    assert np.max(samples) <= 0.1
    assert np.min(samples) >= 0.0000001

    r = loguniform_int(low=1, high=2**12)
    samples = r.rvs(size=1000000)
    assert np.max(samples) == 4096
    assert np.min(samples) == 1

    r = loguniform_int(low=1, high=1000)
    samples = r.rvs(size=1000000)
    assert np.max(samples) == 1000
    assert np.min(samples) == 1

    r = loguniform_int(low=300**0.1, high=300**0.9)
    samples = r.rvs(size=1000000)
    assert np.max(samples) == 170, np.max(samples)
    assert np.min(samples) == 1, np.min(samples)








