import gvar
from gvar import _evalcov_fast as ef
from scipy import sparse
import numpy as np

def evalcov_sparse(g):
    g = np.array(g)
    udata, uindices, uindptr, ldata, lindices, lindptr = ef._evalcov_sparse(g)

    upper = sparse.csr_matrix((udata, uindices, uindptr), shape=2 * (len(g),))
    assert upper.has_canonical_format
    upper = upper.toarray()
    assert np.all(upper - np.triu(upper) == 0)

    lower = sparse.csr_matrix((ldata, lindices, lindptr))
    assert lower.has_canonical_format
    lower = lower.toarray()
    assert np.all(lower - np.tril(lower) == 0)
    
    assert np.array_equal(np.diag(upper), np.diag(lower))
    assert np.array_equal(upper, lower.T)

    cov = upper
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

def compress_labels(l):
    l = np.array(l)
    return list(ef._compress_labels(l))

def test_compress_labels_single_sorted():
    l = np.arange(100)
    idxs = compress_labels(l)
    assert len(idxs) == 1
    assert np.all(idxs[0] == np.arange(len(l)))
    
def test_compress_labels_single():
    l = np.arange(100)
    np.random.shuffle(l)
    idxs = compress_labels(l)
    assert len(idxs) == 1
    assert np.all(idxs[0] == np.arange(len(l)))

def test_compress_labels():
    # labels = np.array([0, 1, 0, 0, 2, 1, 2, 0, 3, 1, 1])
    length = np.random.randint(1, 10, size=10)
    labels = np.concatenate([l * [i] for i, l in enumerate(length)])
    np.random.shuffle(labels)
    indices = np.arange(len(labels))
    idxs = compress_labels(labels)
    assert np.array_equal(np.sort(np.concatenate(idxs)), indices)
    for i in idxs[0]:
        l = labels[i]
        assert np.sum(labels == l) == 1
    for idx in idxs[1:]:
        l = labels[idx[0]]
        assert np.array_equal(idx, indices[labels == l])

def sub_sdev(g, idxs):
    g = np.array(g)
    idxs = np.array(idxs)
    data, indices, indptr, _, _, _ = ef._evalcov_sparse(g)
    return ef._sub_sdev(idxs, data, indices, indptr)

def test_sub_sdev():
    a = np.random.randint(4, size=(10, 10))
    xcov = a.T @ a
    x = gvar.gvar(np.zeros(10), xcov)
    transf = np.random.randint(20, size=(5, 10))
    y = transf @ x
    indices = np.random.randint(len(y), size=5)
    sdev1 = gvar.sdev(y[indices])
    sdev2 = sub_sdev(y, indices)
    assert np.array_equal(sdev1, sdev2)

def sub_cov(g, idxs):
    g = np.array(g)
    idxs = np.array(idxs)
    data, indices, indptr, _, _, _ = ef._evalcov_sparse(g)
    cov = ef._sub_cov(idxs, data, indices, indptr)
    assert cov.shape == (len(idxs), len(idxs))
    return cov

def test_sub_cov():
    a = np.random.randint(5, size=(10, 10))
    xcov = a.T @ a
    x = gvar.gvar(np.zeros(10), xcov)
    transf = np.random.randint(20, size=(5, 10))
    y = transf @ x
    indices = np.random.randint(len(y), size=5)
    cov1 = gvar.evalcov(y[indices])
    cov2 = sub_cov(y, indices)
    assert np.array_equal(cov1, cov2)

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
