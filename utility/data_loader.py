"""
Created on April 10, 2021,
PyTorch Implementation of GNN-based Recommender System
This file is used to read users, items, interaction information
"""
import numpy as np
import os
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


class Data(object):
    def __init__(self, path):
        self.path = path
        self.num_users = 0
        self.num_items = 0
        self.num_entities = 0
        self.num_relations = 0
        self.num_nodes = 0
        self.num_train = 0
        self.num_test = 0

        self.pos_length = None
        self.train_user = None
        self.test_user = None
        self.train_item = None
        self.test_item = None
        self.bipartite_graph = None
        self.user_item_net = None
        self.all_positive = None
        self.test_dict = None
        self.similarity_list = dict()
        self.load_data()

    def load_data(self):
        train_path = self.path + "/train.txt"
        test_path = self.path + "/test.txt"

        print("1.Loading train and test data:")
        print("\t1.1 Loading train dataset:")
        train_user, self.train_user, self.train_item, self.num_train, self.pos_length = self.read_ratings(train_path)
        print("\t\tTrain dataset loading completed.")
        print("\t1.2 Loading test dataset:")
        test_user, self.test_user, self.test_item, self.num_test, _ = self.read_ratings(test_path)
        print("\t\tTest dataset loading completed.")
        print("\tTrain and test dataset loading completed.")

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        self.data_statistics()

        print("2.Construct user-item bipartite graph:")
        assert len(self.train_user) == len(self.train_item)
        self.user_item_net = sp.csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                           shape=(self.num_users, self.num_items))
            
        print("\t Bipartite graph constructed.")

        print("3.Construct adjacency matrix of graph:")

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))
        self.test_dict = self.build_test()


    def sparse_adjacency_matrix_item(self):
        try:
            pre_user_adjacency = sp.load_npz(self.path + '/pre_user_mat.npz')
            print("\t Adjacency matrix of user loading completed.")
            user_adjacency = pre_user_adjacency

            pre_adjacency = sp.load_npz(self.path + '/pre_item_mat.npz')
            print("\t Adjacency matrix of item loading completed.")
            item_adjacency = pre_adjacency

        except:
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.user_item_net.tolil()

            '''
                [ 0  R]
                [R.T 0]
             '''
            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T
            adjacency_matrix = adjacency_matrix.todok()
            
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            # D^(-1/2) A D^(-1/2)
            norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()

            norm_adjacency = norm_adjacency.tolil()
            user_adjacency = norm_adjacency[:self.num_users, self.num_users:]
            user_adjacency = sp.csr_matrix(user_adjacency)
            item_adjacency = norm_adjacency[self.num_users:, :self.num_users]
            item_adjacency = sp.csr_matrix(item_adjacency)
            sp.save_npz(self.path + '/pre_user_mat.npz', user_adjacency)
            sp.save_npz(self.path + '/pre_item_mat.npz', item_adjacency)
            print("\t Adjacency matrix constructed.")

        return user_adjacency, item_adjacency

    def read_ratings(self, file_name):
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)
                if len(pos_id) < 1:
                    print(user_id, pos_id)
                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))
                inter_users.extend([user_id] * len(pos_id))
                pos_length.append(len(pos_id))
                inter_items.extend(pos_id)
                inter_num += len(pos_id)
                line = f.readline()

        return np.array(unique_users), np.array(inter_users), np.array(inter_items), inter_num, pos_length

    def data_statistics(self):
        print("\tnum_users:", self.num_users)
        print("\tnum_items:", self.num_items)
        print("\tnum_nodes:", self.num_nodes)
        print("\tnum_train:", self.num_train)
        print("\tnum_test:", self.num_test)
        print("\tsparisty:", 1 - (self.num_train + self.num_test) / self.num_users / self.num_items)

    def sparse_adjacency_matrix(self):
        if self.bipartite_graph is None:
            try:
                pre_adjacency = sp.load_npz(self.path + '/pre_adj_mat.npz')
                print("\t Adjacency matrix loading completed.")
                norm_adjacency = pre_adjacency
            except:
                adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
                adjacency_matrix = adjacency_matrix.tolil()
                R = self.user_item_net.todok()

                degree_R = R / R.sum(axis=1)
                degree_R[np.isinf(degree_R)] = 0.
                degree_R = sp.csr_matrix(degree_R)
                sp.save_npz(self.path + '/pre_R_mat.npz', degree_R)
                '''
                    [ 0  R]
                    [R.T 0]
                '''
#                 adjacency_matrix[:self.num_users, :self.num_users] = user_matrix
                adjacency_matrix[:self.num_users, self.num_users:] = R
                adjacency_matrix[self.num_users:, :self.num_users] = R.T
                adjacency_matrix = adjacency_matrix.todok()
                # adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

                row_sum = np.array(adjacency_matrix.sum(axis=1))
                d_inv = np.power(row_sum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                degree_matrix = sp.diags(d_inv)

                # D^(-1/2) A D^(-1/2)
                norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
                sp.save_npz(self.path + '/pre_adj_mat.npz', norm_adjacency)
                print("\t Adjacency matrix constructed.")

            self.bipartite_graph = norm_adjacency

        return self.bipartite_graph
    
    def sample_data_to_train_random(self):
        users = np.random.randint(0, self.num_users, len(self.train_user))
        sample_list = []
        for i, user in enumerate(users):
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue
            positive_index = np.random.randint(0, len(positive_items))
            positive_item = positive_items[positive_index]
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)
       
    def sample_data_to_train_all(self):
        sample_list = []
        for i in range(len(self.train_user)):
            user = self.train_user[i]

            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue

            positive_item = self.train_item[i]

            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)
       
    def get_user_pos_items(self, users):
        positive_items = []
        for user in users:
            positive_items.append(self.user_item_net[user].nonzero()[1])
        return positive_items

    def build_test(self):
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
