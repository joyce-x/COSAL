import numba
import numpy as np
import scipy.sparse as sp
import random

numba.set_num_threads(40)

@numba.njit(cache=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon, topk, isSamp, isNorm):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res 
        else:
            p[unode] = res 
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode] and (vnode in q)==False:
                q.append(vnode)
    j_np, val_np = np.array(list(p.keys())), np.array(list(p.values()))
    
    if isSamp:#-----0917
        #weighted sampling
        idx_topk = np.argsort(val_np)[-2*topk:]
        j_np, val_np = j_np[idx_topk], val_np[idx_topk]
        if idx_topk.shape[0] > topk:
            val_np_tmp = val_np.copy()
#             val_np = val_np / np.sum(val_np)
            arr_idx = [idx_ for idx_ in numba.prange(len(j_np))]
            stat = {}
            for j_ in arr_idx:
                stat[j_] = 0
            inode_idx = np.where(j_np==inode)[0][0]
            stat[inode_idx] = 1
            val_np_tmp[inode_idx] = 0
            val_np_tmp = val_np_tmp / np.sum(val_np_tmp)
            for j_ in numba.prange(topk-1):
                sidx = rand_choice_nb(arr_idx, val_np_tmp)
                stat[sidx] = 1
                val_np_tmp[sidx] = 0
                val_np_tmp = val_np_tmp / np.sum(val_np_tmp)
            sample_idx = np.array([sidx for sidx in stat if stat[sidx]==1])
    #         if max(j_np[sample_idx]) > 232964:
    #             print(stat)
    #             print(sample_idx)
    #             print(j_np)
            j_np = j_np[sample_idx] #np.array([j_np[idx] for idx in stat if flag_idx[idx]==True])
            val_np = val_np[sample_idx]
    else:
        idx_topk = np.argsort(val_np)[-topk:]
        j_np, val_np = j_np[idx_topk], val_np[idx_topk]
    
#     if isNorm:#-----0917
#         #normalize
#         val_np = val_np/np.sum(val_np)
# #     return list(p.keys()), list(p.values())
    return j_np, val_np

@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, isSamp, isNorm):#-----0917
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon, topk, isSamp, isNorm)
        js[i] = j
        vals[i] = val
    return js, vals


def ppr_topk( adj_matrix, alpha, epsilon, nodes, topk, isSamp, isNorm):#-----0917
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk, isSamp, isNorm)

    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, isSL, isSamp, isNorm, normalization='row'):#-----0917
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
    if isSL:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
    topk_matrix = ppr_topk(adj_matrix, alpha, eps, idx, topk, isSamp, isNorm).tocsr()#-----0917

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt
        row, col = topk_matrix.nonzero()
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    elif normalization == 'new':
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt
        row, col = topk_matrix.nonzero()
        topk_matrix.data = topk_matrix.data * deg_inv_sqrt[col]
        topk_row_sum = topk_matrix.sum(1).A1
        inv_topk_row_sum = 1/np.maximum(topk_row_sum, 1e-12)
        topk_matrix.data = inv_topk_row_sum[row] * topk_matrix.data
#     elif normalization == 'one':
        
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return topk_matrix
