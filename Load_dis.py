import numpy as np
import multinet
import networkx as nx
import copy
from tkinter import _flatten
from collections import Counter
import math


class load_dis:
    cascade_nodes = []

    def __init__(self, G1, G2, G, node_rank, Node1_cap, Node2_cap, Node_priority, mu, alpha, beta, gamma, theta, Node_attack, Node_handel):
        '''
        :param Node_priority: 节点分配优先级（负载紧迫度）[-，=，+，……]=[1,2,3,4,5……]
        :param G1:
        :param G2:
        :param G:
        :param node_rank: G2节点层级的字典
        :param Node1_cap:子网A的限额系数字典
        :param Node2_cap:子网B的限额系数字典
        :param mu: load adjust
        :param alpha: Parameter
        :param beta:
        :param gamma:
        :param Node_attack:{打击的节点ID:[0/1(失效/低效), int(毁伤百分比)]}
        :param Node_handel:{所有节点ID:int(处理速度，个/秒)}
        '''
        self.G1 = G1
        self.G2 = G2
        self.G = G
        self.node_rank = node_rank
        self.cap1 = Node1_cap
        self.cap2 = Node2_cap
        self.node_inf = {}
        for node in list(self.G.nodes):
            self.node_inf[node] = {}
        self.Node_priority = Node_priority
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.Node_attack = Node_attack
        self.Node_handel = Node_handel
        self.short_neighbour = {}
        self.load_distribution_result = {}

    def g_inf(self):
        '''
        :param node_info: 包含节点ID(所属子网)、层级、内部度和耦合度
        {ID:{node_rank:int,
        inter_degree:int，
        intra_degree:int，
        inter_connect_nodes:[]，
        intra_connect_nodes:[]
        cap_limit:int 容量限额系数}}
        '''
        Intra_num = 0
        for node_1 in list(self.G1.nodes):
            Name_node_1 = 'A' + str(node_1)
            inter_degree = self.G1.in_degree(node_1) + self.G1.out_degree(node_1)  # 出入度和
            intra_degree = self.G.out_degree(Name_node_1) - self.G1.out_degree(node_1)
            Intra_num += intra_degree  # 耦合边数量
            inter_connect_nodes = [i[1] for i in self.G1.out_edges(node_1)]  # 出设边连接节点
            intra_connect_nodes = list(set(m[1] for m in self.G.out_edges(Name_node_1)) - set(
                'A' + str(n[1]) for n in self.G1.out_edges(node_1)))
            cap_limit = self.cap1[node_1]
            self.node_inf[Name_node_1]['node_rank'] = 1
            self.node_inf[Name_node_1]['inter_degree'] = inter_degree
            self.node_inf[Name_node_1]['intra_degree'] = intra_degree
            self.node_inf[Name_node_1]['inter_connect_nodes'] = inter_connect_nodes
            self.node_inf[Name_node_1]['intra_connect_nodes'] = intra_connect_nodes
            self.node_inf[Name_node_1]['cap_limit'] = cap_limit
        for node_2 in list(self.G2.nodes):
            Name_node_2 = 'B' + str(node_2)
            node_rank = self.node_rank[node_2]
            inter_degree = self.G2.in_degree(node_2) + self.G2.out_degree(node_2)
            inter_connect_nodes = [i[1] for i in self.G2.out_edges(node_2)]
            cap_limit = self.cap2[node_2]
            self.node_inf[Name_node_2]['node_rank'] = node_rank
            self.node_inf[Name_node_2]['inter_degree'] = inter_degree
            self.node_inf[Name_node_2]['intra_degree'] = 0
            self.node_inf[Name_node_2]['inter_connect_nodes'] = inter_connect_nodes
            self.node_inf[Name_node_2]['intra_connect_nodes'] = []
            self.node_inf[Name_node_2]['cap_limit'] = cap_limit
        # self.init_and_capacity(Intra_num)  # 本文方法
        self.init_and_capacity3()
        print(self.node_inf)
        print('node_inf ready!!')
        return Intra_num

    def init_and_capacity(self, Num_intra):
        '''
        :param Num_intra: 耦合边数量
        '''
        maxrank = max(self.node_rank.values())
        for node in self.node_inf.keys():
            First = (maxrank * self.mu) / (maxrank - self.node_inf[node]['node_rank'] + 1)
            Second = (self.node_inf[node]['inter_degree'] - self.node_inf[node]['intra_degree']) ** self.alpha
            Third = (self.node_inf[node]['intra_degree'] / Num_intra) ** self.beta
            Fourth = (1 - self.mu) * Third
            node_L0 = round(First * Second + Fourth)
            node_C = self.node_inf[node]['cap_limit'] * (1 + self.gamma) * node_L0
            self.node_inf[node]['Load'] = node_L0
            self.node_inf[node]['Capacity'] = round(node_C)
            self.node_inf[node]['Load_rank_list'] = [1] * node_L0  # 正在处理的负载优先级
            # self.node_inf[node]['Load_delay_list'] = []  # 等待的负载优先级

    def init_and_capacity2(self):
        for node in self.node_inf.keys():
            node_L0 = (self.node_inf[node]['inter_degree'] + self.node_inf[node]['intra_degree']) ** self.alpha
            node_C = (1 + self.gamma) * node_L0
            self.node_inf[node]['Load'] = node_L0
            self.node_inf[node]['Capacity'] = node_C
            self.node_inf[node]['Load_rank_list'] = [1] * node_L0

    def init_and_capacity3(self):
        betw = nx.betweenness_centrality(self.G)
        for node in self.node_inf.keys():
            node_L0 = betw[node] ** self.alpha * 3
            node_C = (1 + self.gamma) * node_L0
            self.node_inf[node]['Load'] = node_L0
            self.node_inf[node]['Capacity'] = node_C
            self.node_inf[node]['Load_rank_list'] = [1] * round(node_L0)

    def determine_delay(self):
        '''
        判断是否进行负载的延迟分配
        :return:
        '''
        All_load = []
        All_cap = []
        flag = False
        for node in self.node_inf:
            All_load.append(self.node_inf[node]['Load'])
            All_cap.append(self.node_inf[node]['Capacity'])
        difference_load = sum(All_load) - sum(All_cap)
        if difference_load > 0:
            flag = True
        return flag, difference_load

    def load_delay(self, Node_attack):
        '''
        更新延迟列表
        :param load: [load1_rank,load2_rank,……]
        :param node_load: 负载分配节点
        :param load_wait_node: 待分配负载节点列表
        :return:负载分配节点的延迟负载
        '''
        Asubnet_load, Asubnet_cap, A_remaining = 0, 0, 0
        Bsubnet_load, Bsubnet_cap, B_remaining = 0, 0, 0
        delay_load = []
        Asub_remain_node, Bsub_remain_node = [], []
        # 得到相应子网的其余所有节点
        for att_node in Node_attack:
            for node in self.node_inf.keys():
                if 'A' == att_node[0] and node != att_node:
                    Asub_remain_node.append(node)
                    Asubnet_load = self.node_inf[node]['Load']
                    Asubnet_cap = self.node_inf[node]['Capacity']
                if 'B' == att_node[0]:
                    Bsub_remain_node.append(node)
                    Bsubnet_load = self.node_inf[node]['Load']
                    Bsubnet_cap = self.node_inf[node]['Capacity']
            # 延迟判断
            A_remaining += Asubnet_load + self.node_inf[att_node]['Load'] - Asubnet_cap
            B_remaining += Bsubnet_load + self.node_inf[att_node]['Load'] - Bsubnet_cap
            # A有延迟
            if A_remaining > 0:
                self.node_inf[att_node]['Load_rank_list'].sort()
                delay_load = self.node_inf[att_node]['Load_rank_list'][:A_remaining]

        return delay_load

    def get_neigbors(self, graph, node, depth=1):
        '''
        计算N阶邻居节点
        :param depth: N
        '''
        output, rank_num = {}, {}
        neighbour_path = []
        layers = dict(nx.bfs_successors(graph, source=node, depth_limit=depth))
        nodes = [node]
        for i in range(1, depth + 1):
            output[i] = []
            for x in nodes:
                output[i].extend([y for y in layers.get(x, []) if y[0] == node[0]])
            nodes = output[i]
        road = list(output.values())
        for length in output:
            if output[length]:
                rank_num[length] = len(output[length])
        path = list(_flatten(road))
        for m in path:
            if m[0] == node[0]:
                neighbour_path.append(m)
        return neighbour_path, rank_num

    def get_roads(self, graph, start_node, end_node, depth=2):
        '''
        除去开始节点和终止节点外，所经过路径上的节点
        :param start_node: 开始节点
        :param end_node: 终止节点
        :param depth: 最大搜索深度
        :return:rank_num：{节点间距：数量}
        '''
        rank_num = {}
        path = []
        neighbour_path = list(nx.all_simple_paths(graph, start_node, end_node, cutoff=depth))
        if neighbour_path:
            neighbour = copy.deepcopy(neighbour_path)
            for m in range(len(neighbour_path)):
                for n in neighbour_path[m]:
                    if n[0] != start_node[0]:
                        neighbour[m].remove(n)
            path = list(set([i for item in neighbour for i in item]))
            path = list(_flatten(path))
            path.remove(start_node)  # 去除出发节点
            # path.remove(end_node)  # 去除目的节点
            for p in sorted(neighbour, key=lambda x:len(x)):
                if len(p) > 1:
                    for node in p:
                        rank_num[node] = rank_num.get(node, len(p)-1)
            del rank_num[start_node]
            rank_num = dict(Counter(rank_num.values()))
        return path, rank_num

    def node_short_neighbour(self, att_node):
        '''
        针对子网AB找到负载分配节点的待分配节点
        :param Node_attack: 受攻击的节点列表
        :return:待分配负载节点信息{受攻击：{待分配节点信息}}
        '''
        flag = True
        G0 = copy.deepcopy(self.G)
        # 目的节点，以及相应的负载待分配节点
        self.short_neighbour[att_node] = {'destination': 0, 'All': [], 'path': 0, 'node_rank': {}, 'rank_num': 0}
        # 目标节点，所有经过节点（待分配负载节点），最大节点间距，{待分配负载节点：节点间距}，{节点间距：数量}（路径从2开始，邻接从1开始）
        rec_node_list = self.node_inf[att_node]['intra_connect_nodes'] # 耦合节点
        res_load = self.node_inf[att_node]['Load'] - self.node_inf[att_node]['Capacity']
        # Network_A
        dis = 0
        if len(rec_node_list) >= 1:
            # 更新待分配负载节点
            self.short_neighbour[att_node]['destination'] = rec_node_list[0]
            # 节点间最短路径排序,计算负载待分配节点，节点路径的扩大
            m = 2
            Atem_capacity = 0 # 节点能力
            # 判断是否加入延迟
            var1 = 1
            path2 = []
            while Atem_capacity < res_load:  # 负载够分
                if var1 > 3:
                    flag = False
                    dis = res_load - Atem_capacity
                    break
                if G0.has_edge(att_node, self.short_neighbour[att_node]['destination']):
                    G0.remove_edge(att_node, self.short_neighbour[att_node]['destination'])
                path, rank_length = self.get_roads(G0, att_node, self.short_neighbour[att_node]['destination'], depth=m)
                # import ipdb;ipdb.set_trace()
                print('第', m-1, '次：', path)
                if path != path2:
                    if path:
                        self.short_neighbour[att_node]['path'] = m
                        self.short_neighbour[att_node]['All'] = path
                        for Anei_cap in self.short_neighbour[att_node]['All']:
                            Atem_capacity += self.node_inf[Anei_cap]['Capacity'] - self.node_inf[Anei_cap]['Load']
                            # 将新添加的路径节点赋予节点间距
                            if not self.short_neighbour[att_node]['node_rank'].get(Anei_cap):
                                self.short_neighbour[att_node]['node_rank'][Anei_cap] = m
                    # 字典rank_length{节点间距：数量}
                    self.short_neighbour[att_node]['rank_num'] = rank_length
                else:
                    var1 += 1
                path2 = copy.deepcopy(path)
                m += 1
        # Network_B
        else:
            n = 1
            Btem_capacity = 0
            var2 = 0
            neighbour2 = []
            while Btem_capacity < res_load:
                if var2 > 3:
                    flag = False
                    dis = res_load - Btem_capacity
                    break
                neighbour, rank_length = self.get_neigbors(G0, att_node, depth=n)
                if neighbour != neighbour2:
                    if neighbour:
                        self.short_neighbour[att_node]['path'] = n
                        self.short_neighbour[att_node]['All'] = neighbour
                        for Bnei_cap in self.short_neighbour[att_node]['All']:
                            Btem_capacity += self.node_inf[Bnei_cap]['Capacity'] - self.node_inf[Bnei_cap]['Load']
                            if not self.short_neighbour[att_node]['node_rank'].get(Bnei_cap):
                                self.short_neighbour[att_node]['node_rank'][Bnei_cap] = n
                    self.short_neighbour[att_node]['rank_num'] = rank_length
                else:
                    var2 += 1
                neighbour2 = copy.deepcopy(neighbour)
                n += 1
        print('攻击', att_node, '节点，待分配负载节点为', self.short_neighbour[att_node]['All'])
        return flag, dis

    def load_redistribution(self, att_node):
        '''
        对待分配负载节点按比例分配
        :param Node_attack: 攻击节点列表
        :param short_neighbour:
        :return:
        '''
        distribution_ratio = {} #{紧迫度差：{'比率':0, 'son':0, 'nodes':[]}}
        self.load_distribution_result[att_node] = {}
        Node_lambda = {}
        First_son, Second_son = {}, {}
        # 全部待分配负载节点中计算第一项，第一项只针对不同层级
        for rec_node in self.short_neighbour[att_node]['All']:
            # 相对层级
            cmp_rank = self.node_inf[rec_node]['node_rank'] - self.node_inf[att_node]['node_rank']
            if not Node_lambda.get(cmp_rank):
                Node_lambda[cmp_rank] = []
            # 该层级下节点的数量
            Node_lambda[cmp_rank].append(self.node_inf[rec_node]['node_rank'])
            if cmp_rank < 0:
                Ratio = 1 / self.Node_priority[-cmp_rank]
            else:
                Ratio = self.Node_priority[cmp_rank]
            if not distribution_ratio.get(cmp_rank):
                # {相对层级：{分配比率，分配结果，该层级下的节点}}
                distribution_ratio[cmp_rank] = {'rank_ratio':0, 'nodes':[]}  # 不重复赋值
                distribution_ratio[cmp_rank]['rank_ratio'] = Ratio
            distribution_ratio[cmp_rank]['nodes'].append(rec_node)
            if not Second_son.get(cmp_rank):
                Second_son[cmp_rank] = {}
            second_son_1 = 1 / self.short_neighbour[att_node]['node_rank'][rec_node]
            second_son_2_1 = self.node_inf[rec_node]['Capacity'] - self.node_inf[rec_node]['Load']
            if second_son_2_1 == 0:
                second_son_2 = 0
            else:
                second_son_2 = second_son_2_1 ** self.theta
            second_son = second_son_1 * second_son_2
            Second_son[cmp_rank][rec_node] = second_son
        for first in Node_lambda.keys():
            First_son[first] = distribution_ratio[first]['rank_ratio'] * len(Node_lambda[first])
        First_mother = sum(First_son.values())
        # 计算
        res_load = self.node_inf[att_node]['Load'] - self.node_inf[att_node]['Capacity']
        for same_rank in list(distribution_ratio.keys()):
            same_rank_load = First_son[same_rank] / First_mother * res_load
            Second_mother = sum(Second_son[same_rank].values())
            # 计算分母后进行同层级下分配
            for node_dis in distribution_ratio[same_rank]['nodes']:
                if Second_mother == 0:
                    result = 0
                else:
                    result = Second_son[same_rank][node_dis] / Second_mother * same_rank_load
                self.load_distribution_result[att_node][node_dis] = round(result)
            # for dis_load in self.load_distribution_result[att_node][node_dis].values():
            #     if dis_load == 0:
            #         result = math.ceil(result)
        # 分配完成后更新本节点Load
        self.node_inf[att_node]['Load'] = self.node_inf[att_node]['Capacity']
        self.node_inf[att_node]['Load_rank_list'].sort()
        self.node_inf[att_node]['Load_rank_list'] = self.node_inf[att_node]['Load_rank_list'][:self.node_inf[att_node]['Capacity']]
        print('低效节点负载分配结果：', self.load_distribution_result[att_node])

    def cmp_load(self, nodes):
        '''
        计算节点集nodes中最小负载，最小容量节点
        :param nodes:
        :return:
        '''
        load_list = {}
        capa_list = {}
        for node in nodes:
            load_list[node] = self.node_inf[node]['Load']
        min_list = []
        result = load_list.values()
        if result:
            min_value = min(result)
            for m, n in load_list.items():
                if n == min_value:
                    min_list.append(m)
            if len(min_list) > 1:
                for double_node in min_list:
                    capa_list[double_node] = self.node_inf[double_node]['Capacity']
                min_list = [min(capa_list, key=capa_list.get)]
        return min_list[0]


    def update(self, att_node):
        '''
        需要更新所有节点Load，延迟列表，分配列表
        :param att_node:
        :param load_res_list:
        :return:
        '''

        # 将延迟分配的节点加入集合, 对节点进行一次更新
        # 整个网络负载分配是否停止
        # remaining = self.node_inf[node]['Capacity'] - self.node_inf[node]['Load'] - len(load)
        # 负载分配节点的延迟列表
        subnet_load, subnet_cap = 0, 0
        # 更新打击节点能力
        print('节点', att_node, '原始节点容量：', self.node_inf[att_node]['Capacity'])
        self.node_inf[att_node]['Capacity'] = math.ceil(self.node_inf[att_node]['Capacity'] * (1 - self.Node_attack[att_node][1]))
        # print('节点打击后剩余容量：', self.node_inf[att_node]['Capacity'])
        original_load = copy.deepcopy(self.node_inf[att_node]['Load'])
        # 计算负载待分配节点
        Flag, dis = self.node_short_neighbour(att_node)
        wait_node_list = self.short_neighbour[att_node]['All']
        node_state = True
        trans_time, fail_load = 0, 0
        for m in wait_node_list:
            if self.Node_attack[m][0] == 1:
                # 待分配负载节点中含低效
                node_state = False
            else:
                node_state = True
        cascade = 0
        if not wait_node_list:
            print('无待分配负载节点，负载丢失,节点', att_node, '需要设置低效态')
            fail_load += self.node_inf[att_node]['Load']
        else:
            while self.node_inf[att_node]['Capacity'] < self.node_inf[att_node]['Load']:  # 需要重分配
                # 更新打击节点
                res_load = self.node_inf[att_node]['Load'] - self.node_inf[att_node]['Capacity']
                print('此时节点需要分配负载：', res_load)
                if res_load > 0 and wait_node_list == []:
                    fail_load += res_load
                    print(fail_load, '负载丢失,节点')
                    break
                if not Flag:  # 无法全部分配，需要判断节点状态
                    if node_state:  # 全为失效节点
                        cascade += 1
                        load_dis.cascade_nodes.append(att_node)
                        # 失效态节点负载分配
                        min_load_node = self.cmp_load(wait_node_list)
                        print('网络发生级联失效', att_node, '节点失效,负载分配至', min_load_node)
                        self.node_inf[min_load_node]['Load'] = self.node_inf[att_node]['Load']
                        self.node_inf[min_load_node]['Load_rank_list'] += self.node_inf[att_node]['Load_rank_list']
                        self.node_inf[att_node]['Capacity'] = 0
                        self.node_inf[att_node]['Load'] = 0
                        self.node_inf[att_node]['Load_rank_list'] = []
                        self.G.remove_node(att_node)
                        del self.node_inf[att_node]
                        wait_node_list.remove(min_load_node)
                        cascade += 1
                        load_dis.cascade_nodes.append(min_load_node)
                        att_node = min_load_node
                        trans_time += 1
                    else:
                        # 含有低效态节点
                        # 将部分负载加入延迟列表
                        if dis > 0:
                            print('网络发生级联失效', att_node, '节点低效,负载延迟分配')
                            self.node_inf[att_node]['Load_rank_list'].sort()
                            All_load = copy.deepcopy(self.node_inf[att_node]['Load_rank_list'])
                            self.node_inf[att_node]['Load_rank_list'] = All_load[dis:]
                            self.node_inf[att_node]['Load'] = len(self.node_inf[att_node]['Load_rank_list'])
                            self.load_redistribution(att_node)
                            self.node_inf[att_node]['Load_rank_list'] += All_load[:dis]
                            self.node_inf[att_node]['Load'] = len(self.node_inf[att_node]['Load_rank_list'])
                            print('延迟后的Load', self.node_inf[att_node]['Load'])
                            Flag, dis = self.node_short_neighbour(att_node)
                            for i, j in self.short_neighbour[att_node]['rank_num'].items():
                                trans_time += (1 + i) * i / 2 * j
                            # self.node_inf[att_node]['Load_delay_list'] = self.node_inf[att_node]['Load_rank_list'][:remaining]
                else:
                    # print('网络负载正常分配！')
                    self.load_redistribution(att_node)
                # 利用节点处理更新所有节点
                self.update_all_node()
                trans_time += 1
                # 删除负载已经分配完的攻击节点
        for all_node in self.node_inf.keys():
            if self.node_inf[all_node]['Load'] > 0:
                self.update_all_node()
                trans_time += 1
        print(cascade, '节点失效', '负载已经全部分完，网络正常运行')
        # print('负载完成消耗时间', trans_time)
        if original_load == 0:
            original_load = 0.000000001
        comp_ratio = 1 - fail_load / original_load
        # print('负载完成率', comp_ratio)
        return cascade, trans_time, comp_ratio
            # if self.node_inf[att_node]['Capacity'] >= self.node_inf[att_node]['Load'] == 0 and self.node_inf[att_node]['Load_rank_list'] == []:
            #     self.G.remove_node(att_node)
            #     del self.node_inf[att_node]

    def attack_cost(self, node_list, cost_alpha, cost_gamma):
        Cost = {}
        for node in node_list:
            Cost[node] = cost_alpha * ((self.node_inf[node]['inter_degree'] + self.node_inf[node]['intra_degree']) ** cost_gamma)
        all_cost = sum(Cost.values())
        return all_cost

    def update_all_node(self):
        for node in self.node_inf.keys():
            if self.node_inf[node]['Capacity'] > 0:
                load_dec = self.node_inf[node]['Load'] - self.Node_handel[node]
                self.node_inf[node]['Load'] = max(load_dec, 0)
                if load_dec <= 0:
                    self.node_inf[node]['Load_rank_list'] = []
                else:
                    self.node_inf[node]['Load_rank_list'].sort()
                    self.node_inf[node]['Load_rank_list'] = self.node_inf[node]['Load_rank_list'][:load_dec]
