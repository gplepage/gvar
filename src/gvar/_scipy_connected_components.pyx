import numpy as np
cimport numpy as np
cimport cython
from numpy cimport npy_intp as ITYPE_t
ITYPE = np.intp

__doc__ = """

This function is copied from scipy/sparse/csgraph/_traversal.pyx. Use it like
this:

labels = np.zeros(csgraph.shape[0], dtype=ITYPE)
n = _connected_components_directed(csgraph.indices, csgraph.indptr, labels)

"""

cdef int _connected_components_directed(
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] indices,
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr,
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] labels):
    """
    Uses an iterative version of Tarjan's algorithm to find the
    strongly connected components of a directed graph represented as a
    sparse matrix (scipy.sparse.csc_matrix or scipy.sparse.csr_matrix).
    The algorithmic complexity is for a graph with E edges and V
    vertices is O(E + V).
    The storage requirement is 2*V integer arrays.
    Uses an iterative version of the algorithm described here:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.102.1707
    For more details of the memory optimisations used see here:
    http://www.timl.id.au/?p=327
    """
    cdef int v, w, index, low_v, low_w, label, j
    cdef int SS_head, root, stack_head, f, b
    DEF VOID = -1
    DEF END = -2
    cdef int N = labels.shape[0]
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] SS, lowlinks, stack_f, stack_b

    lowlinks = labels
    SS = np.ndarray((N,), dtype=ITYPE)
    stack_b = np.ndarray((N,), dtype=ITYPE)
    stack_f = SS

    # The stack of nodes which have been backtracked and are in the current SCC
    SS.fill(VOID)
    SS_head = END

    # The array containing the lowlinks of nodes not yet assigned an SCC. Shares
    # memory with the labels array, since they are not used at the same time.
    lowlinks.fill(VOID)

    # The DFS stack. Stored with both forwards and backwards pointers to allow
    # us to move a node up to the top of the stack, as we only need to visit
    # each node once. stack_f shares memory with SS, as nodes aren't put on the
    # SS stack until after they've been popped from the DFS stack.
    stack_head = END
    stack_f.fill(VOID)
    stack_b.fill(VOID)

    index = 0
    # Count SCC labels backwards so as not to class with lowlinks values.
    label = N - 1
    for v in range(N):
        if lowlinks[v] == VOID:
            # DFS-stack push
            stack_head = v
            stack_f[v] = END
            stack_b[v] = END
            while stack_head != END:
                v = stack_head
                if lowlinks[v] == VOID:
                    lowlinks[v] = index
                    index += 1

                    # Add successor nodes
                    for j in range(indptr[v], indptr[v+1]):
                        w = indices[j]
                        if lowlinks[w] == VOID:
                            with cython.boundscheck(False):
                                # DFS-stack push
                                if stack_f[w] != VOID:
                                    # w is already inside the stack,
                                    # so excise it.
                                    f = stack_f[w]
                                    b = stack_b[w]
                                    if b != END:
                                        stack_f[b] = f
                                    if f != END:
                                        stack_b[f] = b

                                stack_f[w] = stack_head
                                stack_b[w] = END
                                stack_b[stack_head] = w
                                stack_head = w

                else:
                    # DFS-stack pop
                    stack_head = stack_f[v]
                    if stack_head >= 0:
                        stack_b[stack_head] = END
                    stack_f[v] = VOID
                    stack_b[v] = VOID

                    root = 1 # True
                    low_v = lowlinks[v]
                    for j in range(indptr[v], indptr[v+1]):
                        low_w = lowlinks[indices[j]]
                        if low_w < low_v:
                            low_v = low_w
                            root = 0 # False
                    lowlinks[v] = low_v

                    if root: # Found a root node
                        index -= 1
                        # while S not empty and rindex[v] <= rindex[top[S]
                        while SS_head != END and lowlinks[v] <= lowlinks[SS_head]:
                            w = SS_head        # w = pop(S)
                            SS_head = SS[w]
                            SS[w] = VOID

                            labels[w] = label  # rindex[w] = c
                            index -= 1         # index = index - 1
                        labels[v] = label  # rindex[v] = c
                        label -= 1         # c = c - 1
                    else:
                        SS[v] = SS_head  # push(S, v)
                        SS_head = v

    # labels count down from N-1 to zero. Modify them so they
    # count upward from 0
    labels *= -1
    labels += (N - 1)
    return (N - 1) - label