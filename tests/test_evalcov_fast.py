import gvar
from gvar import _evalcov_fast as ef
from scipy import sparse
import numpy as np

def evalcov_sparse(g):
    g = np.array(g)
    data, indices, indptr = ef._evalcov_sparse(g)
    upper = sparse.csr_matrix((data, indices, indptr), shape=2 * (len(g),))
    cov = upper.toarray()
    assert np.all(cov - np.triu(cov) == 0)
    indices = np.triu_indices(len(g))
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
    