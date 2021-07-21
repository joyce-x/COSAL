import torch

from .pytorch_utils import matrix_to_torch
from .utils import OrderedSet
import numpy as np

class PPRDataset(torch.utils.data.Dataset):
    def __init__(self, attr_matrix_all, ppr_matrix, indices, labels_all=None):
        self.attr_matrix_all = attr_matrix_all
        self.ppr_matrix = ppr_matrix
        self.indices = indices
        self.labels_all = torch.tensor(labels_all)
#         self.cached = {}

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # idx is a list of indices
#         key = idx[0]
#         if key not in self.cached:
#         print(idx)
        ppr_matrix = self.ppr_matrix[idx]
#         print(ppr_matrix.shape) #train_num * all_num
        source_idx, neighbor_idx = ppr_matrix.nonzero()
#         print("len(source_idx) = ", len(source_idx)) # train_num * topk
#         x_idx = ([self.indices[idx[i]] for i in source_idx] == neighbor_idx )
        source_true_idx = [self.indices[i] for i in idx]
        nb_idx_set = OrderedSet(np.hstack((neighbor_idx,source_true_idx)))
#         print(len(nb_idx_set)) 
        nb_idx_dict = {ele: i for i,ele in enumerate(nb_idx_set)}
#         print(len(nb_idx_dict))
        nb_idx = [nb_idx_dict[i] for i in neighbor_idx] #index num
#         print((nb_idx))
        x_idx = [nb_idx_dict[self.indices[i]] for i in idx]
#         print(idx)
#         print((x_idx))
#         print(self.indices)
#         print(len(source_idx))

#         ppr_scores = ppr_matrix.data
        ppr_scores = [val for val in ppr_matrix.data if val != 0]

        attr_matrix = matrix_to_torch(self.attr_matrix_all[list(nb_idx_set)])
        ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)
        source_idx = torch.tensor(source_idx, dtype=torch.long)
        x_idx = torch.tensor(x_idx, dtype=torch.long)
        nb_idx = torch.tensor(nb_idx, dtype=torch.long)

        if self.labels_all is None:
            labels = None
        else:
            labels = self.labels_all[self.indices[idx]]
#             self.cached[key] = ((attr_matrix, ppr_scores, source_idx, nb_idx, x_idx), labels)
#         return self.cached[key]
        return ((attr_matrix, ppr_scores, source_idx, nb_idx, x_idx), labels)
