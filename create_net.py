import networkx as nx
import random
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import math


class create_net:
    def __init__(self, n):
        self.n = n

    def prob_value(self, er_p):  # 一个按概率连边的函数
        q = int(10 * er_p)
        l = [1] * q + [0] * (10 - q)
        item = random.sample(l, 1)[0]
        return item

    def er_net(self, er_p):
        into_degree = [0] * self.n  # 节点入度列表
        out_degree = [0] * self.n  # 节点出度列表
        edges = []  # 存储边的列表
        # 拓扑序就按[1,n]的顺序，依次遍历加边
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                if i == 0 and j == self.n - 1:  # 不直连入口和出口
                    continue
                prob = self.prob_value(er_p)  # 连边的概率取0.4
                if prob:
                    if out_degree[i] < 2 and into_degree[j] < 2:  # 限制节点的入度和出度不大于2
                        edges.append((i, j))  # 连边
                        into_degree[j] += 1
                        out_degree[i] += 1
        for node, id in enumerate(into_degree):  # 给所有没有入边的节点添加入口节点作父亲
            if node != 0:
                if id == 0:
                    edges.append((0, node))
                    out_degree[0] += 1
                    into_degree[node] += 1
        for node, od in enumerate(out_degree):  # 给所有没有出边的节点添加出口节点作儿子
            if node != self.n - 1:
                if od == 0:
                    edges.append((node, self.n - 1))
                    out_degree[node] += 1
                    into_degree[self.n - 1] += 1
        # print('边数：', len(edges))
        edges = np.array(edges)
        Adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(self.n, self.n), dtype=int).todense()
        er_g = self.net_generate(Adj)
        return er_g, Adj

    def ws_net(self, edge_link, edge_link_pro):
        p = math.floor(self.n / 2)
        if p % 2 != 0:
            p -= 1
        ws = nx.watts_strogatz_graph(self.n, edge_link, edge_link_pro)
        ws_np = np.array(nx.adjacency_matrix(ws).todense())
        di_ws_np = np.triu(ws_np)
        ws_G = self.net_generate(di_ws_np)
        edges = np.array(ws_G.edges())
        Adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(self.n, self.n), dtype=int).todense()
        return ws_G, Adj

    def ba_net(self, edge_link):
        ba = nx.barabasi_albert_graph(self.n, edge_link)
        ba_np = np.array(nx.adjacency_matrix(ba).todense())
        di_ba_np = np.triu(ba_np)
        ba_G = self.net_generate(di_ba_np)
        edges = np.array(ba_G.edges())
        Adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(self.n, self.n), dtype=int).todense()
        return ba_G, Adj

    def net_generate(self, adj):
        g_type = nx.DiGraph()
        g = nx.from_numpy_matrix(adj, create_using=g_type)
        return g