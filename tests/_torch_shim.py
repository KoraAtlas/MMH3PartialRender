"""Just enough of torch on top of numpy to run test_core.py where torch is unavailable."""
import sys, types
import numpy as np

class T(np.ndarray):
    def __getitem__(self, i):
        return np.asarray(np.asarray(self)[i]).view(T)
    def __array_wrap__(self, arr, context=None, return_scalar=False):
        return np.asarray(arr).view(T)
    @property
    def device(self): return "cpu"
    def to(self, *a, **k):
        for v in list(a) + list(k.values()):
            if isinstance(v, type) or isinstance(v, np.dtype):
                return np.asarray(self).astype(v).view(T)
        return self
    def float(self): return np.asarray(self).astype(np.float32).view(T)
    def clone(self): return np.array(self).view(T)
    def detach(self): return self
    def permute(self, *d): return self.transpose(*d)
    def flatten(self): return np.asarray(self).reshape(-1).view(T)
    def mean(self, dim=None, keepdim=False, **k):
        return np.asarray(np.asarray(self).mean(axis=dim, keepdims=keepdim)).view(T)
    def abs(self): return np.abs(np.asarray(self)).view(T)
    def argmin(self): return np.asarray(self).argmin()
    def clamp(self, min=None, max=None): return np.clip(np.asarray(self), min, max).view(T)
    @property
    def is_nested(self): return False

def _t(a, dtype=None):
    a = np.asarray(a, dtype=dtype)
    return a.view(T)

m = types.ModuleType("torch")
m.as_tensor = lambda a, dtype=None: _t(a, dtype)
m.tensor = lambda a, dtype=None: _t(a, dtype)
m.arange = lambda *a, dtype=None: _t(np.arange(*a), dtype or np.int64)
m.ones = lambda n, dtype=None: _t(np.ones(n), dtype)
m.zeros = lambda n, dtype=None: _t(np.zeros(n), dtype)
m.zeros_like = lambda x: _t(np.zeros_like(np.asarray(x)))
m.ones_like = lambda x: _t(np.ones_like(np.asarray(x)))
m.cat = lambda xs, dim=0: _t(np.concatenate([np.asarray(x) for x in xs], axis=dim))
m.where = lambda c, a, b: _t(np.where(np.asarray(c), np.asarray(a), np.asarray(b)))
m.float32, m.float64, m.long, m.bool = np.float32, np.float64, np.int64, np.bool_
def _interpolate(x, size, mode=None, **k):
    a = np.asarray(x)
    fy, fx = a.shape[2] // size[0], a.shape[3] // size[1]
    if fy * size[0] == a.shape[2] and fx * size[1] == a.shape[3] and fy * fx > 1:
        b, c, h, w = a.shape
        return _t(a.reshape(b, c, size[0], fy, size[1], fx).mean(axis=(3, 5)))
    iy = np.arange(size[0]) * a.shape[2] // size[0]; ix = np.arange(size[1]) * a.shape[3] // size[1]
    return _t(a[:, :, iy][:, :, :, ix])
class _Gen:
    def __init__(self, *a): self.rng = np.random.default_rng(0)
    def manual_seed(self, s): self.rng = np.random.default_rng(s); return self
m.Generator = _Gen
m.randn = lambda shape, generator=None, dtype=None: _t(generator.rng.standard_normal(shape), dtype)
m.nn = types.SimpleNamespace(functional=types.SimpleNamespace(interpolate=_interpolate))
sys.modules["torch"] = m
