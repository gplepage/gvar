import gvar
from gvar import _evalcov_fast as ef

def test_var_primary():
    x = gvar.gvar(0, 123)
    c = ef._cov(x, x)
    assert c == 123 ** 2

def test_cov_primary_indep():
    x = gvar.gvar(0, 2)
    y = gvar.gvar(0, 3)
    c = ef._cov(x, y)
    assert c == 0

def test_cov_primary():
    x, y = gvar.gvar([0, 0], [[1, 0.5], [0.5, 1]])
    c = ef._cov(x, y)
    assert c == 0.5