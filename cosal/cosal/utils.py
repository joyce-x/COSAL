import resource
import numpy as np
import sklearn
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset

from .sparsegraph import load_from_npz

import collections

import os
import networkx as nx
import pickle

from sklearn.metrics import f1_score

import gc

from torch_geometric.utils.convert import to_scipy_sparse_matrix

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.shape[1]]

        return sp.csr_matrix((data, indices, indptr), shape=shape)


def split_random(seed, n, n_train, n_val):
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx

def get_data_ogb(data_dir,data_name):
    dataset = PygNodePropPredDataset(root = data_dir, name = data_name) 

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]
    
#     attr_matrix = graph.__getitem__('x')
    num_node = graph.num_nodes
    
    edge_index = graph.__getitem__("edge_index")
    row_index = edge_index[0].numpy()
    col_index = edge_index[1].numpy()
    num_edge = graph.num_edges
    data = np.array([1 for i in range(num_edge)])
    
    adj_matrix = sp.csr_matrix((data,(row_index,col_index)), shape=(num_node, num_node))
    
    return adj_matrix, graph.x, graph.y.numpy().flatten(), train_idx.numpy(), valid_idx.numpy(), test_idx.numpy()

def split_train_val(seed, train_idx, n_train):
    np.random.seed(seed)
    n = train_idx.shape[0]
    rnd = np.random.permutation(n)

    train_idx_ = train_idx[np.sort(rnd[:n_train])]
    if(2*n_train >= n):
        val_idx_ = train_idx[np.sort(rnd[n_train:n])]
        test_idx_ = np.array([])
        return train_idx_, val_idx_, test_idx_
    else:
        val_idx_ = train_idx[np.sort(rnd[n_train:n_train*2])]
        test_idx_ = train_idx[np.sort(rnd[n_train*2 : n])]
        return train_idx_, val_idx_, test_idx_

def get_data_ogb_v2(data_dir, data_name, seed, ntrain_div_classes):
    dataset = PygNodePropPredDataset(root = data_dir, name = data_name) 

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"].numpy(), split_idx["valid"].numpy(), split_idx["test"].numpy()
    
    num_classes = dataset.num_classes
    graph = dataset[0]
    
    del dataset
    gc.collect()
    
    #filter out unlabeled test nodes
    labels = graph.y.numpy().flatten()
    is_labeled = labels[:]==labels[:]
    test_idx = [idx for idx in test_idx if is_labeled[idx]]
    
    # split train, val, test newly
    test_idx = np.hstack((valid_idx, test_idx))
    n_train = ntrain_div_classes * num_classes
    train_idx, valid_idx, test_idx_ = split_train_val(seed, train_idx, n_train)
    test_idx = np.hstack((test_idx_, test_idx)).astype(valid_idx.dtype) 
    
#     attr_matrix = graph.__getitem__('x')

#     num_node = graph.num_nodes
#     edge_index = graph.__getitem__("edge_index")
#     row_index = edge_index[0].numpy()
#     col_index = edge_index[1].numpy()
#     num_edge = graph.num_edges
#     data = np.array([1 for i in range(num_edge)])
#     adj_matrix = sp.csr_matrix((data,(row_index,col_index)), shape=(num_node, num_node))

    adj_matrix = to_scipy_sparse_matrix(graph.edge_index).tocsr()
    
    return adj_matrix, graph.x.numpy(), labels.astype(int), train_idx, valid_idx, test_idx, num_classes

def get_data(dataset_path, seed, ntrain_div_classes):
    '''
    Get data from a .npz-file.

    Parameters
    ----------
    dataset_path
        path to dataset .npz file
    seed
        Random seed for dataset splitting
    ntrain_div_classes
        Number of training nodes divided by number of classes
    

    '''
    g = load_from_npz(dataset_path)

    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()

    # number of nodes and attributes
    n, d = g.attr_matrix.shape

#     # optional attribute normalization
#     if normalize_attr == 'per_feature':
#         if sp.issparse(g.attr_matrix):
#             scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
#         else:
#             scaler = sklearn.preprocessing.StandardScaler()
#         attr_matrix = scaler.fit_transform(g.attr_matrix)
#     elif normalize_attr == 'per_node':
#         if sp.issparse(g.attr_matrix):
#             attr_norms = sp.linalg.norm(g.attr_matrix, ord=1, axis=1)
#             attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
#             attr_matrix = g.attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
#         else:
#             attr_norms = np.linalg.norm(g.attr_matrix, ord=1, axis=1)
#             attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
#             attr_matrix = g.attr_matrix * attr_invnorms[:, np.newaxis]
#     else:
#         attr_matrix = g.attr_matrix

    # helper that speeds up row indexing
    if sp.issparse(g.attr_matrix):
        attr_matrix = SparseRowIndexer(g.attr_matrix)
    else:
        attr_matrix = g.attr_matrix

    # split the data into train/val/test
    num_classes = g.labels.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 10
    train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)


    return g.adj_matrix, attr_matrix, g.labels, train_idx, val_idx, test_idx

def get_data_multilabel(dataset_path, seed, ntrain_div_classes, normalize_attr=None):
    '''
    Get data from a .npz-file.

    Parameters
    ----------
    dataset_path
        path to dataset .npz file
    seed
        Random seed for dataset splitting
    ntrain_div_classes
        Number of training nodes divided by number of classes
    normalize_attr
        Normalization scheme for attributes. By default (and in the paper) no normalization is used.

    '''
    g = load_from_npz(dataset_path)

    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()

    # number of nodes and attributes
    n, d = g.attr_matrix.shape

    # optional attribute normalization
    if normalize_attr == 'per_feature':
        if sp.issparse(g.attr_matrix):
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        attr_matrix = scaler.fit_transform(g.attr_matrix)
    elif normalize_attr == 'per_node':
        if sp.issparse(g.attr_matrix):
            attr_norms = sp.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
        else:
            attr_norms = np.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix * attr_invnorms[:, np.newaxis]
    else:
        attr_matrix = g.attr_matrix

    # helper that speeds up row indexing
    if sp.issparse(attr_matrix):
        attr_matrix = SparseRowIndexer(attr_matrix)
    else:
        attr_matrix = attr_matrix

    # split the data into train/val/test
    num_classes = g.labels.shape[1]
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 10
    train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)


    return g.adj_matrix, attr_matrix, g.labels, train_idx, val_idx, test_idx

def get_data_(dataset_path, seed, ntrain_div_classes, normalize_attr=None):
    '''
    Get data from a .npz-file.

    Parameters
    ----------
    dataset_path
        path to dataset .npz file
    seed
        Random seed for dataset splitting
    ntrain_div_classes
        Number of training nodes divided by number of classes
    normalize_attr
        Normalization scheme for attributes. By default (and in the paper) no normalization is used.

    '''
    g = load_from_npz(dataset_path)

    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()

    # number of nodes and attributes
    n, d = g.attr_matrix.shape

    # optional attribute normalization
    if normalize_attr == 'per_feature':
        if sp.issparse(g.attr_matrix):
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        attr_matrix = scaler.fit_transform(g.attr_matrix)
    elif normalize_attr == 'per_node':
        if sp.issparse(g.attr_matrix):
            attr_norms = sp.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
        else:
            attr_norms = np.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix * attr_invnorms[:, np.newaxis]
    else:
        attr_matrix = g.attr_matrix

    # helper that speeds up row indexing
#     if sp.issparse(attr_matrix):
#         attr_matrix = SparseRowIndexer(attr_matrix)
#     else:
#         attr_matrix = attr_matrix

    # split the data into train/val/test
    num_classes = g.labels.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 10
    train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)


    return g.adj_matrix, attr_matrix, g.labels, train_idx, val_idx, test_idx


def get_data_v2(data_name, data_dir):
    # Reading the data...
    tmp = []
    prefix = os.path.join(data_dir, 'ind.%s.' % data_name)
    for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
        with open(prefix + suffix, 'rb') as fin:
            tmp.append(pickle.load(fin, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tmp
    with open(prefix + 'test.index') as fin:
        tst_idx = [int(i) for i in fin.read().split()]
    assert np.sum(x != allx[:x.shape[0], :]) == 0
    assert np.sum(y != ally[:y.shape[0], :]) == 0

    # Spliting the data...
    trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
    val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
    tst_idx = np.array(tst_idx, dtype=np.int64)
    assert len(trn_idx) == x.shape[0]
    assert len(trn_idx) + len(val_idx) == allx.shape[0]
    assert len(tst_idx) > 0
    assert len(set(trn_idx).intersection(val_idx)) == 0
    assert len(set(trn_idx).intersection(tst_idx)) == 0
    assert len(set(val_idx).intersection(tst_idx)) == 0

    # Building the graph...
    graph = nx.from_dict_of_lists(graph)
    assert min(graph.nodes()) == 0
    n = graph.number_of_nodes()
    assert max(graph.nodes()) + 1 == n
    n = max(n, np.max(tst_idx) + 1)
    for u in range(n):
        graph.add_node(u)
    assert graph.number_of_nodes() == n
    assert not graph.is_directed()
    adj_matrix = nx.to_scipy_sparse_matrix(graph)

    # Building the feature matrix and the label matrix...
    d, c = x.shape[1], y.shape[1]
    feat_ridx, feat_cidx, feat_data = [], [], []
    allx_coo = allx.tocoo()
    for i, j, v in zip(allx_coo.row, allx_coo.col, allx_coo.data):
        feat_ridx.append(i)
        feat_cidx.append(j)
        feat_data.append(v)
    tx_coo = tx.tocoo()
    for i, j, v in zip(tx_coo.row, tx_coo.col, tx_coo.data):
        feat_ridx.append(tst_idx[i])
        feat_cidx.append(j)
        feat_data.append(v)
    if data_name.startswith('nell.0'):
        isolated = np.sort(np.setdiff1d(range(allx.shape[0], n), tst_idx))
        for i, r in enumerate(isolated):
            feat_ridx.append(r)
            feat_cidx.append(d + i)
            feat_data.append(1)
        d += len(isolated)
    feat = sp.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
    targ = np.zeros((n, c), dtype=np.int64)
    targ[trn_idx, :] = y
    targ[val_idx, :] = ally[val_idx, :]
    targ[tst_idx, :] = ty
    targ = dict((i, j) for i, j in zip(*np.where(targ)))
    targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
    #print('#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c))
    
    return adj_matrix, feat, targ, trn_idx, val_idx, tst_idx

def get_data_v3(data_file, seed, ntrain_div_classes):
#     start = time.time()

    with np.load(data_file, allow_pickle=True) as loader:
        loader = dict(loader)
#     print(loader)

    # adj_data = loader['adj_matrix.data']
    # adj_indices = loader['adj_matrix.indices']
    # adj_indptr = loader['adj_matrix.indptr']
    # adj_shape = loader['adj_matrix.shape']
    # attr_data = loader['attr_matrix.data']
    # attr_indices = loader['attr_matrix.indices']
    # attr_indptr = loader['attr_matrix.indptr']
    # attr_shape = loader['attr_matrix.shape']
    # labels = loader['labels']
    # class_names = loader['class_names']

    if (data_file == 'data/reddit.npz' or data_file == 'data/amazon2M.npz'):
        attr_matrix = sp.csr_matrix(loader['attr_matrix'])
        adj_matrix = sp.csr_matrix((loader['adj_data'],loader['adj_indices'],loader['adj_indptr']),shape=loader['adj_shape'])
    else:
        attr_matrix = sp.csr_matrix((loader['attr_matrix.data'],loader['attr_matrix.indices'],loader['attr_matrix.indptr']),shape=loader['attr_matrix.shape'])
        adj_matrix = sp.csr_matrix((loader['adj_matrix.data'],loader['adj_matrix.indices'],loader['adj_matrix.indptr']),shape=loader['adj_matrix.shape'])
    labels = loader['labels']
    # split the data into train/val/test
    n, d = attr_matrix.shape
    num_classes = labels.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 10
    train_idx, val_idx, test_idx = split_random(seed, n, n_train, n_val)

    return adj_matrix, attr_matrix, labels, train_idx, val_idx, test_idx

def get_max_memory_bytes():
    return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def calc_new_ppr_score(ppr_matrix, sim_matrix, args):
    sim_score = sim_matrix.data
#     ppr_score = [val for val in ppr_matrix.data if val != 0]
    ppr_score = ppr_matrix.data
    syn_score = (map(lambda x : x[0] * args.gamma + x[1] * (1 - args.gamma), zip(ppr_score, sim_score)))
    syn_score = np.fromiter(syn_score, dtype=np.float32)
    return syn_score

##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))
        
def get_inf_nodes(seed, test_idx, n_test):
    np.random.seed(seed)
    rnd = np.random.permutation(test_idx)
    test_idx_inf = np.sort(rnd[:n_test])
    return test_idx_inf

def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")

def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro