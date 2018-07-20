""" Basic linear algebra for GVars. """

# Created by G. Peter Lepage (Cornell University) on 2014-04-27.
# Copyright (c) 2015-2018 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy
import gvar

def det(a):
    """ Determinant of matrix ``a``.

    Args:
        a: Two-dimensional, square matrix/array of numbers
            and/or :class:`gvar.GVar`\s.

    Returns:
        Deterimant of the matrix.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('bad matrix shape: ' + str(a.shape))
    da = a - amean
    ainv = inv(amean)
    return numpy.linalg.det(amean) * (1 + numpy.matrix.trace(da.dot(ainv)))

def slogdet(a):
    """ Sign and logarithm of determinant of matrix ``a``.

    Args:
        a: Two-dimensional, square matrix/array of numbers
            and/or :class:`gvar.GVar`\s.

    Returns:
        Tuple ``(s, logdet)`` where the determinant of matrix ``a`` is
            ``s * exp(logdet)``.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('bad matrix shape: ' + str(a.shape))
    da = a - amean
    ainv = inv(amean)
    s, ldet = numpy.linalg.slogdet(amean)
    ldet += numpy.matrix.trace(da.dot(ainv))
    return s, ldet

def eigvalsh(a, eigvec=False):
    """ Eigenvalues of Hermitian matrix ``a``.

    Args:
        a: Two-dimensional, square Hermitian matrix/array of numbers
            and/or :class:`gvar.GVar`\s. Array elements must be
            real-valued if `gvar.GVar`\s are involved (i.e., symmetric
            matrix).
        eigvec (bool): If ``True``, method returns a tuple of arrays
            ``(val, vec)`` where ``val[i]`` are the
            eigenvalues of ``a``, and ``vec[:, i]`` are the mean
            values of the corresponding eigenvectors. Only ``val`` is
            returned if ``eigvec=False`` (default).

    Returns:
        Array ``val`` of eigenvalues of matrix ``a`` if parameter
        ``eigvec==False`` (default); otherwise a tuple of
        arrays ``(val, vec)`` where ``val[i]`` are the eigenvalues
        (in ascending order) and ``vec[:, i]`` are the mean values
        of the corresponding eigenvectors.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    if eigvec == True:
        val, vec = eigh(a, eigvec=True)
        return val, gvar.mean(vec)
    else:
        return eigh(a, eigvec=False)


def eigh(a, eigvec=True, rcond=None):
    """ Eigenvalues and eigenvectors of symmetric matrix ``a``.

    Args:
        a: Two-dimensional, square Hermitian matrix/array of numbers
            and/or :class:`gvar.GVar`\s. Array elements must be
            real-valued if `gvar.GVar`\s are involved (i.e., symmetric
            matrix).
        eigvec (bool): If ``True``  (default), method returns a tuple
            of arrays ``(val, vec)`` where ``val[i]`` are the
            eigenvalues of ``a`` (in ascending order), and ``vec[:, i]``
            are the corresponding eigenvectors of ``a``. Only ``val`` is
            returned if ``eigvec=False``.
        rcond (float): Eigenvalues whose difference is smaller than
            ``rcond`` times their sum are assumed to be degenerate
            (and ignored) when computing variances for the eigvectors.
            Default (``rcond=None``) is ``max(M,N)`` times machine precision.

    Returns:
        Tuple ``(val,vec)`` of eigenvalues and eigenvectors of
        matrix ``a`` if parameter ``eigvec==True`` (default).
        The eigenvalues ``val[i]`` are in ascending order and
        ``vec[:, i]`` are the corresponding eigenvalues. Only
        the eigenvalues ``val`` are returned if ``eigvec=False``.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    a = numpy.asarray(a)
    if a.dtype != object:
        val, vec = numpy.linalg.eigh(a)
        return (val, vec) if eigvec else val
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('bad matrix shape: ' + str(a.shape))
    if rcond is None:
        rcond = numpy.finfo(float).eps * max(a.shape)
    da = a - amean
    val0, vec0 = numpy.linalg.eigh(amean)
    val = val0 + [
        vec0[:, i].conjugate().dot(da.dot(vec0[:, i])) for i in range(vec0.shape[1])
        ]
    if eigvec == True:
        if vec0.dtype == complex:
            raise ValueError('cannot evaluate eigenvectors when a is complex')
        vec = numpy.array(vec0, dtype=object)
        for i in range(len(val)):
            for j in range(len(val)):
                dval = val0[i] - val0[j]
                if abs(dval) < rcond * abs(val0[j] + val0[i]) or dval == 0.0:
                    continue
                vec[:, i] += vec0[:, j] * (
                    vec0[:, j].dot(da.dot(vec0[:, i])) / dval
                    )
        return val, vec
    else:
        return val

def svd(a, compute_uv=True, rcond=None):
    """ svd decomposition of matrix ``a`` containing |GVar|\s.

    Args:
        a: Two-dimensional matrix/array of numbers
            and/or :class:`gvar.GVar`\s.
        compute_uv (bool): It ``True`` (default), returns
            tuple ``(u,s,vT)`` where matrix ``a = u @ np.diag(s) @ vT``
            where matrices ``u`` and ``vT`` satisfy ``u.T @ u = 1``
            and ``vT @ vT.T = 1``, and ``s`` is the list of singular
            values. Only ``s`` is returned if ``compute_uv=False``.
        rcond (float): Singular values whose difference is smaller than
            ``rcond`` times their sum are assumed to be degenerate for
            calculating variances for ``u`` and ``vT``.
            Default (``rcond=None``) is ``max(M,N)`` times machine precision.

    Returns:
        Tuple ``(u,s,vT)`` where matrix ``a = u @ np.diag(s) @ vT``
        where matrices ``u`` and ``vT`` satisfy ``u.T @ u = 1``
        and ``vT @ vT.T = 1``, and ``s`` is the list of singular
        values. If ``a.shape=(N,M)``, then ``u.shape=(N,K)``
        and ``vT.shape=(K,M)`` where ``K`` is the number of
        nonzero singular values (``len(s)==K``).
        If ``compute_uv==False`` only ``s`` is returned.

    Raises:
        ValueError: If matrix is not two-dimensional.
    """
    a = numpy.asarray(a)
    if a.dtype != object:
        return numpy.linalg.svd(a, compute_uv=compute_uv)
    amean = gvar.mean(a)
    if amean.ndim != 2:
        raise ValueError(
            'matrix must have dimension 2: actual shape = ' + str(a.shape)
            )
    if rcond is None:
        rcond = numpy.finfo(float).eps * max(a.shape)
    da = a - amean
    u0,s0,v0T = numpy.linalg.svd(amean, compute_uv=True, full_matrices=True)
    k = min(a.shape)
    s = s0 + [
        u0[:, i].dot(da.dot(v0T[i, :])) for i in range(k)
        ]
    if compute_uv:
        u = numpy.array(u0, dtype=object)
        vT = numpy.array(v0T, dtype=object)
        # u first
        daaT = da.dot(a.T) + a.dot(da.T)
        s02 = numpy.zeros(daaT.shape[0], float)
        s02[:len(s0)] = s0 ** 2
        for j in range(s02.shape[0]):
            for i in range(k):
                if i == j:
                    continue
                ds2 = s02[i]  - s02[j]
                if abs(ds2) < rcond * abs(s02[i] + s02[j]) or ds2 == 0:
                    continue
                u[:, i] +=  u0[:, j]  * u0[:, j].dot(daaT.dot(u0[:, i])) / ds2
        # v next
        daTa = da.T.dot(a) + a.T.dot(da)
        s02 = numpy.zeros(daTa.shape[0], float)
        s02[:len(s0)] = s0 ** 2
        for j in range(s02.shape[0]):
            for i in range(k):
                if i == j:
                    continue
                ds2 = s02[i]  - s02[j]
                if abs(ds2) < rcond * abs(s02[i] + s02[j]) or ds2 == 0:
                    continue
                vT[i, :] +=  v0T[j, :]  * v0T[j, :].dot(daTa.dot(v0T[i, :])) / ds2
        return u[:,:k], s, vT[:k, :]
    else:
        return s


def lstsq(a, b, rcond=None, weighted=False, extrainfo=False):
    """ Least-squares solution ``x`` to ``a @ x = b`` for |GVar|\s.

    Here ``x`` is defined to be the solution that minimizes ``||b - a @ x||``.
    If ``b`` has a covariance matrix, another option is to weight the
    norm with the inverse covariance matrix: i.e., minimize
    ``|| isig @ b - isig @ a @ x||`` where ``isig`` is the square root of the
    inverse of ``b``'s covariance matrix. Set parameter ``weighted=True`` to
    obtain the weighted-least-squares solution.

    Args:
        a : Matrix/array of shape ``(M,N)`` containing numbers and/or |GVar|\s.
        b : Vector/array of shape ``(M,)`` containing numbers and/or |GVar|\s.
        rcond (float): Cutoff for singular values of ``a``. Singular values
            smaller than ``rcond`` times the maximum eigenvalue are ignored.
            Default (``rcond=None``) is ``max(M,N)`` times machine precision.
        weighted (bool): If ``True``, use weighted least squares; otherwise
            use unweighted least squares.
        extrainfo (bool): If ``False`` (default) only ``x`` is returned;
            otherwise ``(x, residual, rank, s)`` is returned.
    Returns:
        Array ``x`` of shape ``(N,)`` that minimizes ``|| b - a @ x||``
        if ``extrainfo==False`` (default); otherwise returns a tuple
        ``(x, residual, rank, s)`` where ``residual`` is the sum
        of the squares of ``b - a @ x``, ``rank`` is the rank of matrix
        ``a``, and ``s`` is an array containing the singular values.
    """
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    if a.ndim != 2:
        raise ValueError(
            'a must have dimension 2: actual shape = ' + str(a.shape)
            )
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            'a and b shapes mismatched: {} vs {}'.format(a.shape, b.shape)
            )
    if rcond is None:
        rcond = numpy.finfo(float).eps * max(a.shape)
    if weighted:
        try:
            cov = gvar.evalcov(b)
        except ValueError:
            raise ValueError('b does not have a covariance matrix')
        try:
            icov = numpy.linalg.inv(cov)
        except numpy.linalg.LinAlgError:
            raise ValueError("b's covariance matrix cannot be inverted")
        ata = a.T.dot(icov.dot(a))
        atb = a.T.dot(icov.dot(b))
    else:
        ata = a.T.dot(a)
        atb = a.T.dot(b)
    val, vec = gvar.linalg.eigh(ata)
    maxval = numpy.max(gvar.mean(val))  # N.B. val > 0 required
    ans = 0
    for i in range(len(val)):
        if gvar.mean(val[i]) < rcond * maxval:
            continue
        ans += vec[:, i] * vec[:, i].dot(atb) / val[i]
    if not extrainfo:
        return ans
    val = val[val >= rcond * maxval] ** 0.5
    d = a.dot(ans) - b
    residual = d.dot(icov.dot(d)) if weighted else d.dot(d)
    k = len(val)
    return ans, residual, k, val


def inv(a):
    """ Inverse of matrix ``a``.

    Args:
        a: Two-dimensional, square matrix/array of numbers
            and/or :class:`gvar.GVar`\s.

    Returns:
        The inverse of matrix ``a``.

    Raises:
        ValueError: If matrix is not square and two-dimensional.
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('bad matrix shape: ' + str(a.shape))
    da = a - amean
    ainv = numpy.linalg.inv(amean)
    return ainv - ainv.dot(da.dot(ainv))

def solve(a, b):
    """ Find ``x`` such that ``a @ x = b`` for matrix ``a``.

    Args:
        a: Two-dimensional, square matrix/array of numbers
            and/or :class:`gvar.GVar`\s.
        b: One-dimensional vector/array of numbers and/or
            :class:`gvar.GVar`\s, or an array of such vectors.
            Requires ``b.shape[0] == a.shape[1]``.

    Returns:
        The solution ``x`` of ``a.dot(x) = b``, which is equivalent
        to ``inv(a).dot(b)``.

    Raises:
        ValueError: If ``a`` is not square and two-dimensional.
        ValueError: If shape of ``b`` does not match that of ``a``
            (that is ``b.shape[0] != a.shape[1]``).
    """
    amean = gvar.mean(a)
    if amean.ndim != 2 or amean.shape[0] != amean.shape[1]:
        raise ValueError('bad matrix shape: ' + str(a.shape))
    bmean = gvar.mean(b)
    if bmean.shape[0] != a.shape[1]:
        raise ValueError(
            'Mismatch between shapes of a and b: {} {}'.format(a.shape, b.shape)
            )
    # xmean = numpy.linalg.solve(amean, bmean)
    ainv = numpy.linalg.inv(amean)
    xmean = ainv.dot(bmean)
    return xmean + ainv.dot(b-bmean - (a-amean).dot(xmean))
