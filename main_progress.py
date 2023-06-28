import numpy as np
import random
import networkx as nx
from Load_dis import load_dis
import copy
from multinet import mutilayer_net
from create_net import create_net
import math
import json
import time
from multiprocessing import Pool
import sys
import powerlaw


def aver_min_max(value):
    value = list(value)
    max_value = max(value)
    min_value = min(value)
    if len(value) > 2:
        value.remove(max_value)
        if min_value in value:
            value.remove(min_value)
    if not value:
        aver_value = 0
    else:
        aver_value = np.mean(value)
    return aver_value


def main(G, G1, G2, p1, p2, alpha):
    # sum_list = []
    pro_gress = percolation(G, G1, G2, p1, p2, alpha)
    graph = copy.deepcopy(G)  # G.to_undirected()
    AB = pro_gress.progress(graph)
    # sum_list.append(len(AB))
    return len(AB)


class percolation:
    # 1.建立网络模型，同时模拟渗流过程，分析pc和S
    # 2.重分配结果后，分析pc和S
    def __init__(self, network, network_A, network_B, p1, p2, alpha):
        self.Network = network
        self.graph_A = network_A
        self.graph_B = network_B
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha
        # self.node_inf = {}
        # for node in list(self.Network.nodes):
        #     self.node_inf[node] = {}

    def g_inf(self):
        for node_1 in list(self.graph_A.nodes):
            Name_node_1 = 'A' + str(node_1)
            inter_degree = self.graph_A.in_degree(node_1) + self.graph_A.out_degree(node_1)  # 出入度和
            intra_degree = self.Network.out_degree(Name_node_1) - self.graph_A.out_degree(node_1)
            self.node_inf[Name_node_1]['inter_degree'] = inter_degree
            self.node_inf[Name_node_1]['intra_degree'] = intra_degree
            self.node_inf[Name_node_1]['cost'] = (inter_degree + intra_degree) ** 2
        for node_2 in list(self.graph_B.nodes):
            Name_node_2 = 'B' + str(node_2)
            inter_degree = self.graph_B.in_degree(node_2) + self.graph_B.out_degree(node_2)
            self.node_inf[Name_node_2]['inter_degree'] = inter_degree
            self.node_inf[Name_node_2]['intra_degree'] = 0
            self.node_inf[Name_node_2]['cost'] = inter_degree ** 2

    def attack_nodes_class(self):
        # 蓄意攻击(驱动攻击)时，攻击节点按照子网络A、B进行分类
        A_attack_nodes, B_attack_nodes = [], []
        # todo 需要修改
        for node in load_dis.att_nodes:
            if node[0] == 'A':
                A_attack_nodes.append(node)
            else:
                B_attack_nodes.append(node)
        return A_attack_nodes, B_attack_nodes

    def attack_nodes(self):
        # 渗流模型下，按比例设计(随机选择)攻击节点
        A_nodes = list(self.graph_A.nodes)
        B_nodes = list(self.graph_B.nodes)
        A_attack_num = round(self.p1 * len(A_nodes))
        B_attack_num = round(self.p2 * len(B_nodes))
        A_attack_nodes = random.sample(A_nodes, A_attack_num)
        B_attack_nodes = random.sample(B_nodes, B_attack_num)
        return A_attack_nodes, B_attack_nodes

    def delib_attack_nodes(self):
        # 度攻击
        A_nodes = sorted(self.graph_A.degree, key=lambda x: x[1], reverse=True)
        B_nodes = sorted(self.graph_B.degree, key=lambda x: x[1], reverse=True)
        A_attack_num = round(self.p1 * len(A_nodes))
        B_attack_num = round(self.p2 * len(B_nodes))
        A_attack_nodes = [i[0] for i in A_nodes][:A_attack_num]
        B_attack_nodes = [i[0] for i in B_nodes][:B_attack_num]
        return A_attack_nodes, B_attack_nodes

    def between_attack_nodes(self):
        # 介数攻击
        A_nodes = sorted(nx.betweenness_centrality(self.graph_A).items(), key=lambda x: x[1], reverse=True)
        B_nodes = sorted(nx.betweenness_centrality(self.graph_B).items(), key=lambda x: x[1], reverse=True)
        A_attack_num = round(self.p1 * len(A_nodes))
        B_attack_num = round(self.p2 * len(B_nodes))
        A_attack_nodes = [i[0] for i in A_nodes][:A_attack_num]
        B_attack_nodes = [i[0] for i in B_nodes][:B_attack_num]
        return A_attack_nodes, B_attack_nodes

    def Cost_cal(self, A_nodes, B_nodes):
        Cost = 0  # 脆弱节点集代价
        for a_dri_node in A_nodes:
            Cost += (self.graph_A.in_degree(int(a_dri_node)) + self.graph_A.out_degree(int(a_dri_node))) ** 0.5
            # (self.Network.in_degree('A' + str(a_dri_node)) + self.Network.out_degree('A' + str(a_dri_node))) ** 0.5
        for b_dri_node in B_nodes:
            Cost += (self.graph_B.in_degree(int(b_dri_node)) + self.graph_B.out_degree(int(b_dri_node))) ** 0.5
        return Cost

    def cost_number_attack(self):
        A_dri_nodes, B_dri_nodes = self.driver_attack_nodes()
        Nodes = (len(A_dri_nodes) + len(B_dri_nodes))
        ratio = Nodes / 500
        return ratio

    def cost_attack_nodes(self):
        # 得到相同代价下的需要攻击的节点
        Cost = 0  # 脆弱节点集代价
        A_dri_nodes, B_dri_nodes = self.driver_attack_nodes()
        for a_dri_node in A_dri_nodes:
            Cost += (self.graph_A.in_degree(int(a_dri_node)) + self.graph_A.out_degree(int(a_dri_node))) ** 0.5
                #(self.Network.in_degree('A' + str(a_dri_node)) + self.Network.out_degree('A' + str(a_dri_node))) ** 0.5
        for b_dri_node in B_dri_nodes:
            Cost += (self.graph_B.in_degree(int(b_dri_node)) + self.graph_B.out_degree(int(b_dri_node))) ** 0.5
                #(self.Network.in_degree('B' + str(b_dri_node)) + self.Network.out_degree('B' + str(b_dri_node))) ** 0.5
        A_attack_nodes, B_attack_nodes = self.delib_attack_nodes()
        # A_attack_nodes, B_attack_nodes = self.between_attack_nodes()
        # 同比例攻击节点代价
        Wait_cost, Wait_node = [], []
        for a_node in A_attack_nodes:
            a_node_cost = (self.graph_A.in_degree(int(a_node)) + self.graph_A.out_degree(int(a_node))) ** 0.5
            # W_Cost += a_node_cost
            Wait_cost.append(a_node_cost)
            Wait_node.append('A' + str(a_node))
        for b_node in B_attack_nodes:
            b_node_cost = (self.graph_B.in_degree(int(b_node)) + self.graph_B.out_degree(int(b_node))) ** 0.5
            # W_Cost += b_node_cost
            Wait_cost.append(b_node_cost)
            Wait_node.append('B' + str(b_node))
        # c = list(zip(Wait_node, Wait_cost))
        # random.shuffle(c)
        # Wait_node, Wait_cost = zip(*c)
        Cost_index = np.argsort(-np.array(Wait_cost))
        index, tem_cha = 0, 0
        if Cost_index.size == 0:
            attack_nodes_A, attack_nodes_B = [], []
        else:
            flag = False
            while not flag:
                if index >= len(Cost_index):
                    index = len(Cost_index)
                    flag = True
                else:
                    tem_cha += Wait_cost[Cost_index[index]]
                    if tem_cha >= Cost:
                        flag = True
                index += 1
            result_cost = [n for i, n in enumerate(Wait_cost) if i in Cost_index[:index-1]]
            result_nodes = [n for i, n in enumerate(Wait_node) if i in Cost_index[:index-1]]
            attack_nodes_A, attack_nodes_B = [], []
            for node in result_nodes:
                if node[0] == 'A':
                    attack_nodes_A.append(int(node[1:]))
                if node[0] == 'B':
                    attack_nodes_B.append(int(node[1:]))
        return attack_nodes_A, attack_nodes_B

    def progress(self, graph):
        current_S = [0]
        # 渗流下攻击节点
        attack_nodes_A, attack_nodes_B = self.attack_nodes()
        # attack_nodes_A, attack_nodes_B = self.delib_attack_nodes()
        # attack_nodes_A, attack_nodes_B = self.driver_attack_nodes()
        # attack_nodes_A, attack_nodes_B = self.between_attack_nodes()
        #相同数量下攻击节点
        # attack_ratio = self.cost_number_attack()
        # attack_nodes_A = attack_nodes_A[:math.floor(len(attack_nodes_A) * attack_ratio)]
        # attack_nodes_B = attack_nodes_B[:math.floor(len(attack_nodes_B) * attack_ratio)]
        # 相同代价下攻击节点
        # attack_nodes_A, attack_nodes_B = self.cost_attack_nodes()
        # print(attack_nodes_A, attack_nodes_B)
        # todo 重分配后攻击节点
        # A_attack_nodes, B_attack_nodes = self.attack_nodes_class()
        A_attack_nodes = ['A%s' % i for i in attack_nodes_A]
        B_attack_nodes = ['B%s' % j for j in attack_nodes_B]
        A_del_nodes = copy.deepcopy(A_attack_nodes)
        B_del_nodes = copy.deepcopy(B_attack_nodes)
        # BFS删除多阶邻居
        Graph = copy.deepcopy(graph)
        # B_nodes_list = [i for i in Graph.nodes if i[0] == 'B']
        # Graph.remove_nodes_from(B_nodes_list)
        flag = True
        S_A, S_B = 0, 0
        while flag:
            Graph, A_del_nodes = self.BFS_delete(Graph, A_attack_nodes, A_del_nodes, 'A')
            # 耦合节点\alpha失效
            ori_A_B_nodes = []
            for A_B_node in A_del_nodes:
                intra_node = [i for i in list(self.Network[A_B_node]) if i[0] == 'B']
                ori_A_B_nodes.extend(intra_node)
            A_B_nodes = random.sample(ori_A_B_nodes, math.ceil(self.alpha * len(ori_A_B_nodes)))
            B_del_nodes.extend(A_B_nodes)
            Graph.remove_nodes_from(A_B_nodes)

            # 对网络B操作(todo 网络B再影响网络A循环)
            Graph, B_del_nodes = self.BFS_delete(Graph, B_attack_nodes, B_del_nodes, 'B')
            # 耦合节点\alpha失效
            ori_B_A_nodes = self.Network_B_intra(B_del_nodes)
            B_A_nodes = random.sample(ori_B_A_nodes, math.ceil(self.alpha * len(ori_B_A_nodes)))
            A_del_nodes.extend(B_A_nodes)
            Graph.remove_nodes_from(B_A_nodes)
            S, S_A, S_B = self.find_clus(Graph)
            if len(current_S) == len(S):
                # print('序参量:', len(S))
                flag = False
            else:
                current_S = copy.deepcopy(S)
                # print('循环', current_S)
        # print('节点数量:', len(Graph.nodes))
        return current_S, S_A, S_B

    def Network_B_intra(self, B_nodes_list):
        und_G = self.Network.to_undirected()
        B_A_nodes = []
        for B_node in B_nodes_list:
            B_A_nodes.extend([node for node in list(und_G[B_node]) if node[0] == 'A'])
        return B_A_nodes

    def BFS_delete(self, graph, attack_nodes, del_nodes, network_type):
        queue = copy.deepcopy(attack_nodes)
        while queue:
            current_node = queue.pop(0)
            if current_node not in graph:
                # print('跳过节点：', current_node)
                continue
            current_node_neighbour = list(graph[current_node])
            inter_neighbour = [m for m in current_node_neighbour if m[0] == network_type]
            del_p = math.ceil(len(inter_neighbour) / 1)
            inter_del_neighbour = random.sample(inter_neighbour, del_p)
            queue.extend(inter_del_neighbour)
            del_nodes.extend(inter_del_neighbour)
            # print(len(del_nodes))
            graph.remove_nodes_from(del_nodes)
        return graph, list(set(del_nodes))

    def find_clus(self, last_graph):
        und_G = last_graph.to_undirected()
        a_nodes = [i for i in und_G.nodes if i[0] == 'A']
        b_nodes = [i for i in und_G.nodes if i[0] == 'B']
        large_A, large_B = {}, {}
        if len(list(und_G.nodes)) == 0:
            large = {}
        else:
            und_a_G = copy.deepcopy(und_G)
            und_b_G = copy.deepcopy(und_G)
            und_a_G.remove_nodes_from(b_nodes)
            und_b_G.remove_nodes_from(a_nodes)
            large = max(nx.connected_components(und_G), key=len)
            if len(list(und_a_G.nodes)) != 0 and len(list(und_b_G.nodes)) != 0:
                large_A = max(nx.connected_components(und_a_G), key=len)
                large_B = max(nx.connected_components(und_b_G), key=len)
            elif len(list(und_b_G.nodes)) == 0:
                large_A = max(nx.connected_components(und_a_G), key=len)
                large_B = {}
            elif len(list(und_a_G.nodes)) == 0:
                large_B = max(nx.connected_components(und_b_G), key=len)
                large_A = {}
        # isolate_nodes = set(und_G.nodes) - large
        # new_G = last_graph.remove_nodes_from(list(isolate_nodes))
        return large, large_A, large_B