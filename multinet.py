import networkx as nx
import matplotlib.pyplot as plt
import create_net as cn
import random
import numpy as np
import matplotlib.patheffects as path_effects


# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
class mutilayer_net:

    # def __init__(self, Num_1, Num_2):
    #     self.node_1 = cn.create_net(Num_1)
    #     self.node_2 = cn.create_net(Num_2)

    def twolayer_net(self, G1, G2):
        # G1, er_adj = self.node_1.er_net(self.er_p)
        # G2, ba_adj = self.node_2.ba_net(self.ba_l)
        G = nx.union(G1, G2, rename=('A', 'B'))
        subnet_connect = random.choices(list(G2.nodes), k=len(G1))
        link, link2 = [], []
        edge = []
        for i, j in zip(G1.nodes, subnet_connect):
            link.append(('A' + str(i), 'B' + str(j)))
            # link2.append(('B' + str(j), 'A' + str(i)))
            edge.append((i, j))
        G.add_edges_from(link)
        # 双向连边
        # G.add_edges_from(link2)
        # self.mul_net_plot(link, G)
        return G1, G2, G, edge