import gvar
from gvar import _evalcov_fast as ef
from scipy import sparse
import numpy as np

def evalcov_sparse(g):
    g = np.array(g)
    ldata, lindices, lindptr = ef._evalcov_sparse(g)

    lower = sparse.csr_matrix((ldata, lindices, lindptr))
    assert lower.has_canonical_format
    lower = lower.toarray()
    assert np.all(lower - np.tril(lower) == 0)
    
    cov = lower
    indices = np.tril_indices(len(g))
    cov[tuple(reversed(indices))] = cov[indices]
    assert np.all(cov == cov.T)
    return cov

def test_var_primary():
    x = gvar.gvar(0, 123)
    cov = evalcov_sparse([x, x])[0, 0]
    assert cov == 123 ** 2

def test_cov_primary_indep():
    x = gvar.gvar(0, 2)
    y = gvar.gvar(0, 3)
    cov = evalcov_sparse([x, y])
    assert np.all(cov == [[4, 0], [0, 9]])

def test_cov_primary():
    c = [[1, 0.5], [0.5, 1]]
    x, y = gvar.gvar([0, 0], c)
    cov = evalcov_sparse([x, y])
    assert np.all(cov == c)

def test_cov():
    a = np.random.randint(4, size=(10, 10))
    xcov = a.T @ a
    x = gvar.gvar(np.zeros(10), xcov)

    cov1 = evalcov_sparse(x)
    cov2 = gvar.evalcov(x)
    assert np.all(cov1 == cov2)
    
    transf = np.random.randint(4, size=(5, 10))
    y = transf @ x
    
    cov1 = evalcov_sparse(y)
    cov2 = gvar.evalcov(y)
    assert np.all(cov1 == cov2)

def test_empty_cov():
    x = gvar.gvar(0, 0)
    out = ef.evalcov_blocks(x)
    assert np.array_equal(out[0][0], [0])
    assert np.array_equal(out[0][1], [[0.]])

def test_evalcov_blocks():
    def test_cov(g):
        if hasattr(g, 'keys'):
            g = gvar.BufferDict(g)
        g = g.flat[:]
        cov = np.zeros((len(g), len(g)), dtype=float)
        for idx, bcov in ef.evalcov_blocks(g):
            cov[idx[:,None], idx] = bcov
        assert str(gvar.evalcov(g)) == str(cov)
    g = gvar.gvar(5 * ['1(1)'])
    test_cov(g)
    g[-1] = g[0] + g[1]
    test_cov(g)
    test_cov(g * gvar.gvar('2(1)'))
    g = gvar.gvar(5 * ['1(1)'])
    g[0] = g[-1] + g[-2]
    test_cov(g)

def test_evalcov_blocks_compress():
    def test_cov(g):
        if hasattr(g, 'keys'):
            g = gvar.BufferDict(g)
        blocks = ef.evalcov_blocks(g, compress=True)
        g = g.flat[:]
        cov = np.zeros((len(g), len(g)), dtype=float)
        idx, bsdev = blocks[0]
        if len(idx) > 0:
            cov[idx, idx] = bsdev ** 2
        for idx, bcov in blocks[1:]:
            cov[idx[:,None], idx] = bcov
        assert str(gvar.evalcov(g)) == str(cov)
    g = gvar.gvar(5 * ['1(1)'])
    test_cov(g)
    test_cov(dict(g=g))
    g[-1] = g[0] + g[1]
    test_cov(g)
    test_cov(dict(g=g))
    test_cov(g * gvar.gvar('2(1)'))
    g = gvar.gvar(5 * ['1(1)'])
    g[0] = g[-1] + g[-2]
    test_cov(g)
    test_cov(dict(g=g))
    g[1:] += g[:-1]
    test_cov(g)
    test_cov(dict(g=g))
