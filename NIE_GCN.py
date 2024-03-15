"""
Created on April 16, 2021,
PyTorch Implementation of GNN-based Recommender System
This file is used to construct NIE-GCN
"""
import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
from utility.data_loader import Data


class NIE_GCN(nn.Module):
    def __init__(self, config, dataset: Data, device):
        super(NIE_GCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.count = 10
        self.showtime = 0
        
        self.user_embedding = None
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=self.config.dim)
        
        initializer = nn.init.xavier_uniform_
        
        self.weight_dict = nn.ParameterDict()

        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        user_R, item_R = self.dataset.sparse_adjacency_matrix_item()
        
        self.user_R = self.convert_sp_mat_to_sp_tensor(user_R)
        self.user_R = self.user_R.coalesce().to(self.device)

        self.item_R = self.convert_sp_mat_to_sp_tensor(item_R)
        self.item_R = self.item_R.coalesce().to(self.device)
        

        self.attention_dense = nn.Sequential(
            nn.Linear(self.config.dim * 2, self.config.dim),
            nn.Tanh(),
            nn.Linear(self.config.dim, 1, bias=False)
        )
        
        self.activation = nn.Sigmoid()
        self.activation_layer = nn.Tanh()
        self.attention_activation = nn.ReLU()
        
    def convert_sp_mat_to_sp_tensor(self, sp_mat):
        """
            coo.row: x in user-item graph
            coo.col: y in user-item graph
            coo.data: [value(x,y)]
        """
        coo = sp_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        value = torch.FloatTensor(coo.data)
        # from a sparse matrix to a sparse float tensor
        sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
        return sp_tensor

    def aggregate(self):

        item_embedding = self.item_embedding.weight

        all_item_embeddings = []
        all_user_embeddings = []
        
        # aggregate

        for layer in range(self.config.GCNLayer):
            
            user_embedding = self.activation_layer(torch.sparse.mm(self.user_R, item_embedding))
            item_embedding = self.activation_layer(torch.sparse.mm(self.item_R, user_embedding))

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)

        if self.config.agg == 'cat':
            final_user_embeddings = torch.cat(all_user_embeddings, dim=1)
            final_item_embeddings = torch.cat(all_item_embeddings, dim=1)
        elif self.config.agg == "sum":
            final_user_embeddings = torch.stack(all_user_embeddings, dim=1)
            final_user_embeddings = torch.sum(final_user_embeddings, dim=1)

            final_item_embeddings = torch.stack(all_item_embeddings, dim=1)
            final_item_embeddings = torch.sum(final_item_embeddings, dim=1)
        
        return final_user_embeddings, final_item_embeddings

    
    def get_bpr_loss(self, user, positive, negative):

        all_user_embeddings, all_item_embeddings = self.aggregate()
    
        user_embedding = all_user_embeddings[user.long()]
        
        positive_embedding = all_item_embeddings[positive.long()]
        negative_embedding = all_item_embeddings[negative.long()]
        
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        reg_loss = (1 / 2) * (ego_pos_emb.norm(2).pow(2) + ego_neg_emb.norm(2).pow(2)) / float(len(user))

        pos_score = torch.sum(torch.mul(user_embedding, positive_embedding), dim=1)

        neg_score = torch.sum(torch.mul(user_embedding, negative_embedding), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        reg_loss = reg_loss * self.config.l2

        return loss, reg_loss
    
    def get_rating_for_test(self, user):

        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        
        return rating
    
    def update_attention_A(self):
        self.user_embedding = torch.sparse.mm(self.user_R, self.item_embedding.weight)

        fold_len = len(self.dataset.train_user) // self.count
        attention_score = []

        for i in range(self.count):
            start = i * fold_len
            if i == self.count - 1:
                end = len(self.dataset.train_user)
            else:
                end = (i + 1) * fold_len

            A_socre = self.attention_score(self.dataset.train_user[start:end], self.dataset.train_item[start:end])

            attention_score.append(A_socre)

        attention_score = torch.cat(attention_score).squeeze()

        new_attention_score = attention_score.detach().cpu()
        
        new_attention_score = np.exp(new_attention_score)

        new_R = sp.coo_matrix((new_attention_score, (self.dataset.train_user, self.dataset.train_item)), shape=(self.dataset.num_users, self.dataset.num_items))

        new_R = new_R / np.power(new_R.sum(axis=1), self.config.beta)

        new_R[np.isinf(new_R)] = 0.

        new_R = sp.csr_matrix(new_R)
        
        self.user_R = self.user_R.cpu()
        
        torch.cuda.empty_cache()

        del self.user_R

        self.user_R = self.convert_sp_mat_to_sp_tensor(new_R).coalesce().to(self.device)

        
    def attention_score(self, users, items):
        assert len(users) == len(items)
        users = torch.Tensor(users).to(self.device)
        items = torch.Tensor(items).to(self.device)
        user_embedding = self.user_embedding[users.long()]
        item_embedding = self.item_embedding(items.long())
#         norm_user_embedding = user_embedding / user_embedding.norm(dim=0, keepdim=True)
#         norm_item_embedding = item_embedding / item_embedding.norm(dim=0, keepdim=True)
#         score = (norm_user_embedding * norm_item_embedding).sum(dim=1) ** 2
        
        embedding = nn.functional.relu(torch.cat([user_embedding, item_embedding], dim=1))
        score = self.attention_dense(embedding)

        return score.squeeze()    