import random
from typing import List

import env
import scipy.optimize as op
import numpy as np
import time
import datetime

real_graph = []

def generate_tuopu(edge: List[env.Edge]):
    if (len(real_graph) == len(edge)):
        return real_graph
    real_graph.clear()
    num = len(edge)
    for _ in range(num):
        real_graph.append([1000 for _ in range(num)])
    for i in range(num):
        if i > 0:
            real_graph[i][i-1] = 1
            real_graph[i-1][i] = 1
            for j in range(i-1):
                if random.randint(1, 100) > 50:
                    real_graph[i][j] = 1
                    real_graph[j][i] = 1
    for i in range(num):
        real_graph[i][i] = 0
    return real_graph


def Dijsktra(graph: np.array, start: int):
    num = len(graph)
    num = len(graph)
    shortPath = [0 for _ in range(num)]
    visited = [0 for _ in range(num)]
    visited[start] = 1
    shortPath[start] = 0
    for _ in range(num-1):
        k = -1
        dmin = 1000
        for i in range(num):
            if visited[i] == 0 and graph[start][i] < dmin:
                dmin = graph[start][i]
                k = i
        shortPath[k] = dmin
        visited[k] = 1
        for i in range(num):
            if visited[i] == 0 and graph[start][k] + graph[k][i] < graph[start][i]:
                graph[start][i] = graph[start][k] + graph[k][i]
    return shortPath

def cal_transmission(mobile: List[env.Mobile], edge: List[env.Edge], environment: env.Environment):
    graph = generate_tuopu(edge=edge)
    shortPath = []
    for i in range(len(edge)):
        shortPath.append(Dijsktra(graph=graph, start=i))
    transmission_time = []
    for _ in range(len(edge)):
        transmission_time.append([0 for _ in range(len(mobile))])
    for i in range(len(edge)):
        for j in range(len(mobile)):
            datasize = mobile[j].datasize
            trans_rate = environment.trans_rate(mobile[j].trans_power, mobile[j].distance)
            trans_time = environment.trans_time(trans_rate, datasize)
            if i != mobile[j].connect_edge:
                trans_time += (environment.trans_time(environment.fibic_network, datasize) + 0.025) * shortPath[mobile[j].connect_edge][i]
            transmission_time[i][j] = trans_time
    return transmission_time


def cal_workload(mobile: List[env.Mobile], edge: List[env.Edge], environment: env.Environment) -> np.array:
    # 计算移动设备的任务卸载到各个边缘服务器需要的计算资源
    workload = []
    transmission_time = cal_transmission(mobile, edge, environment)
    # print(transmission_time)
    for _ in range(len(edge)):
        workload.append([0 for _ in range(len(mobile))])
    for i in range(len(edge)):
        for j in range(len(mobile)):
            compute_load = env.compute_size[mobile[j].task_type]
            trans_time = transmission_time[i][j]
            # print(trans_time)
            if trans_time >= env.max_latency:
                workload[i][j] = 10000
            else:
                workload[i][j] = compute_load / (env.max_latency-trans_time)
    # print(workload)
    return workload


def mobile_belong(mobile: List[env.Mobile]):
    mobile_service = []
    task_num = [0 for _ in range(env.service_num)]
    for _ in range(len(mobile)):
        mobile_service.append([0 for _ in range(env.service_num)])
    for j in range(len(mobile)):
        service_id = mobile[j].task_type
        mobile_service[j][service_id] = 1
        task_num[service_id] += 1
    return mobile_service, task_num


class Result_info:
    def __init__(self, throughput):
        self.throughput = throughput
        self.opt = None
        self.average_latency = None

    def get_min_ratio(self):
        return sum(self.throughput)

    def set_opt(self, opt):
        self.opt = opt

    def get_opt(self):
        return self.opt

    # def set_average_latency(self, latency):
    #     self.average_latency = latency

    # def get_average_latency(self):
    #     return sum(self.average_latency) / len(self.average_latency)


class my_algo():
    def __init__(self, mobile: List[env.Mobile], edge: List[env.Edge], environment: env.Environment) -> None:
        self.mobile = mobile
        self.edge = edge
        self.environment = environment
        self.workload = cal_workload(self.mobile, self.edge, self.environment)
        self.mobile_service, self.task_num = mobile_belong(self.mobile)
        self.mobile_num = len(self.mobile)
        self.edge_num = len(self.edge)
        self.init_idx()
        self.throughput = Throughput(mobile, edge, environment, self.workload)

    def re_init(self, mobile: List[env.Mobile], edge: List[env.Edge], environment: env.Environment) -> None:
        self.mobile = mobile
        self.edge = edge
        self.environment = environment
        self.workload = cal_workload(self.mobile, self.edge, self.environment)
        self.mobile_service, self.task_num = mobile_belong(self.mobile)
        self.mobile_num = len(self.mobile)
        self.edge_num = len(self.edge)
        self.init_idx()
        self.throughput.re_init(self.mobile, self.edge, self.environment, self.workload)

    def init_idx(self):
        self.x_start = 0
        self.x_length = self.edge_num * env.service_num
        self.y_start = self.x_start + self.x_length
        self.y_length = self.edge_num * self.mobile_num
        self.g_start = self.y_start + self.y_length
        self.g_length = 1
        self.total_length = self.g_start + self.g_length

    def runner(self) -> dict:
        result = {}
        t1 = int(round(time.time() * 1000))
        result_info = self.approximate_algorithm()
        t2 = int(round(time.time() * 1000))
        result["appro"] = result_info
        result["t1"] = t2 - t1
        # result_info = self.determin_rounding()
        # result["determin_rounding"] = result_info
        result_info1, result_info2, timeCost1, timeCost2 = self.max_throughput()
        result["max_thro"] = result_info1
        result["t2"] = timeCost1
        result["max_thro-greedy"] = result_info2
        result["t3"] = timeCost2
        t1 = int(round(time.time() * 1000))
        result_info = self.greedy()
        t2 = int(round(time.time() * 1000))
        result["greedy"] = result_info
        result["t4"] = t2 - t1
        return result


    def true_runner(self) -> dict:
        result = {}
        t1 = int(round(time.time() * 1000))
        result_info = self.approximate_algorithm()
        t2 = int(round(time.time() * 1000))
        result["true_appro"] = result_info
        result["t5"] = t2 - t1
        # result_info = self.determin_rounding()
        # result["determin_rounding"] = result_info
        t1 = int(round(time.time() * 1000))
        result_info = self.greedy()
        t2 = int(round(time.time() * 1000))
        result["true_greedy"] = result_info
        result["t6"] = t2 - t1
        return result

    def approximate_algorithm(self):
        res = self.lr_max_ratio()
        # print(res.fun)
        fra_placement = res.x[self.x_start: self.x_start + self.x_length]
        fra_placement = np.reshape(fra_placement, (self.edge_num, env.service_num))
        placement_strategy = self.round_placement_strategy(fra_placement.tolist())
        y = res.x[self.y_start: self.y_start + self.y_length]
        y = np.reshape(y, (self.edge_num, self.mobile_num))
        fra_offloading = np.zeros(np.shape(y))
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                service_id = self.mobile[m].task_type
                if placement_strategy[s][service_id] != 0:
                    fra_offloading[s][m] = y[s][m] / fra_placement[s][service_id]
                else:
                    fra_offloading[s][m] = 0
        offloading_strategy = self.round_offloading_strategy(fra_offloading.tolist())
        placement_strategy = self.throughput.service_placement_strategy_check(placement_strategy, fra_placement)
        self.throughput.check_placement_strategy(placement_strategy)
        offloading_strategy = self.throughput.modify_offloading_by_correct_edge(placement_strategy, offloading_strategy)
        fra_offloading = np.zeros(np.shape(y))
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                service_id = self.mobile[m].task_type
                if placement_strategy[s][service_id] != 0:
                    fra_offloading[s][m] = y[s][m] / fra_placement[s][service_id]
                else:
                    fra_offloading[s][m] = 0
        offloading_strategy = self.throughput.offloading_strategy_check_computing_constraints(placement_strategy, offloading_strategy, fra_offloading)
        offloading_strategy = self.throughput.offloading_strategy_non_overlapping(offloading_strategy, fra_offloading)
        offloading_strategy = self.throughput.offloading_strategy_max_ratio(placement_strategy, offloading_strategy)
        self.throughput.check_offloading_strategy(placement_strategy, offloading_strategy)
        ratio, _ = self.throughput.generate_ratio(offloading_strategy)
        self.throughput.generate_ratio_print(offloading_strategy)
        return ratio
    
    def determin_rounding(self):
        res = self.lr_max_ratio()
        # print(res.fun)
        fra_placement = res.x[self.x_start: self.x_start + self.x_length]
        fra_placement = np.reshape(fra_placement, (self.edge_num, env.service_num))
        placement_strategy = fra_offloading = np.zeros(np.shape(fra_placement))
        for s in range(self.edge_num):
            for n in range(env.service_num):
                if fra_offloading[s][n] >= 0.5:
                    placement_strategy[s][n] = 1
                else:
                    placement_strategy[s][n] = 0
        placement_strategy = self.throughput.service_placement_strategy_check(placement_strategy, fra_placement)
        self.throughput.check_placement_strategy(placement_strategy)
        fra_offloading = res.x[self.y_start: self.y_start + self.y_length]
        fra_offloading = np.reshape(fra_offloading, (self.edge_num, self.mobile_num))
        offloading_strategy = self.round_offloading_strategy(fra_offloading.tolist())
        offloading_strategy = self.throughput.modify_offloading_by_correct_edge(placement_strategy, offloading_strategy)
        offloading_strategy = self.throughput.offloading_strategy_non_overlapping(offloading_strategy, fra_offloading)
        offloading_strategy = self.throughput.offloading_strategy_check_computing_constraints(offloading_strategy, fra_offloading)
        self.throughput.check_offloading_strategy(placement_strategy, offloading_strategy)
        ratio, _ = self.throughput.generate_ratio(offloading_strategy)
        return ratio

    def max_throughput(self):
        t1 = int(round(time.time() * 1000))
        res = self.lr_max_throughput()
        # print(res.fun)
        fra_placement = res.x[self.x_start: self.x_start + self.x_length]
        fra_placement = np.reshape(fra_placement, (self.edge_num, env.service_num))
        placement_strategy = self.round_placement_strategy(fra_placement.tolist())
        y = res.x[self.y_start: self.y_start + self.y_length]
        y = np.reshape(y, (self.edge_num, self.mobile_num))
        fra_offloading = np.zeros(np.shape(y))
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                service_id = self.mobile[m].task_type
                if placement_strategy[s][service_id] != 0:
                    fra_offloading[s][m] = y[s][m] / fra_placement[s][service_id]
                else:
                    fra_offloading[s][m] = 0
        offloading_strategy = self.round_offloading_strategy(fra_offloading.tolist())
        placement_strategy = self.throughput.service_placement_strategy_check_static(placement_strategy, fra_placement)
        self.throughput.check_placement_strategy(placement_strategy)
        offloading_strategy = self.throughput.modify_offloading_by_correct_edge(placement_strategy, offloading_strategy)
        fra_offloading = np.zeros(np.shape(y))
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                service_id = self.mobile[m].task_type
                if placement_strategy[s][service_id] != 0:
                    fra_offloading[s][m] = y[s][m] / fra_placement[s][service_id]
                else:
                    fra_offloading[s][m] = 0
        offloading_strategy = self.throughput.offloading_strategy_check_computing_constraints(placement_strategy, offloading_strategy, fra_offloading)
        offloading_strategy = self.throughput.offloading_strategy_non_overlapping(offloading_strategy, fra_offloading)
        t2 = int(round(time.time() * 1000))
        ratio1, _ = self.throughput.generate_ratio(offloading_strategy)
        self.throughput.generate_ratio_print(offloading_strategy)
        offloading_strategy = self.throughput.offloading_strategy_max_throughput(placement_strategy, offloading_strategy)
        t3 = int(round(time.time() * 1000))
        self.throughput.check_offloading_strategy(placement_strategy, offloading_strategy)
        ratio2, _ = self.throughput.generate_ratio(offloading_strategy)
        self.throughput.generate_ratio_print(offloading_strategy)
        return ratio1, ratio2, t2 - t1, t3 - t1

    def greedy(self):
        graph = generate_tuopu(edge=self.edge)
        shortPath = []
        for i in range(len(self.edge)):
            shortPath.append(Dijsktra(graph=graph, start=i))
        path = []
        for i in range(len(self.edge)):
            path.append(np.argsort(shortPath[i]))
        placement_strategy = []
        offloading_strategy = []
        for _ in range(self.edge_num):
            placement_strategy.append([0 for _ in range(env.service_num)])
        for _ in range(self.edge_num):
            offloading_strategy.append([0 for _ in range(self.mobile_num)])
        storage_load = [0 for _ in range(self.edge_num)]
        compute_load = [0 for _ in range(self.edge_num)]
        offloading_count = np.sum(offloading_strategy, axis=0)
        while 1 > 0:
            n = self.throughput.generate_ratio_random(offloading_strategy)
            flag = 0
            for m in range(self.mobile_num):
                if offloading_count[m] == 0:
                    ids = self.mobile[m].connect_edge
                    if self.mobile[m].task_type == n:
                        for s in path[ids]:
                            if placement_strategy[s][n] == 1:
                                if compute_load[s] + self.workload[s][m] <= self.edge[s].computing_capacity:
                                    offloading_strategy[s][m] = 1
                                    compute_load[s] += self.workload[s][m]
                                    offloading_count[m] = 1
                                    flag = 1
                                    break
                        if flag == 1:
                            break
            if flag == 0:
                for m in range(self.mobile_num):
                    if offloading_count[m] == 0:
                        ids = self.mobile[m].connect_edge
                        if self.mobile[m].task_type == n:
                            for s in path[ids]:
                                if placement_strategy[s][n] == 0:
                                    if compute_load[s] + self.workload[s][m] <= self.edge[s].computing_capacity:
                                        if storage_load[s] + env.service_size[n] <= self.edge[s].storage_capacity:
                                            placement_strategy[s][n] = 1
                                            offloading_strategy[s][m] = 1
                                            offloading_count[m] = 1
                                            compute_load[s] += self.workload[s][m]
                                            storage_load[s] += env.service_size[n]
                                            flag = 1
                                            break
                    if flag == 1:
                        break
            if flag == 0:
                break
        self.throughput.check_placement_strategy(placement_strategy)
        self.throughput.check_offloading_strategy(placement_strategy, offloading_strategy)
        ratio, _ = self.throughput.generate_ratio(offloading_strategy)
        self.throughput.generate_ratio_print(offloading_strategy)
        return ratio


    def round_placement_y(self, placement_strategy: List[np.array], y: List[np.array]) -> np.array:
        fra_offloading_strategy = np.zeros(np.shape(y))
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                service_id = self.mobile[m].task_type
                fra_offloading_strategy[s][m] = y[s][m] / placement_strategy[s][service_id]
        return fra_offloading_strategy

    def round_placement_strategy(self, placement_strategy: List[np.array]) -> np.array:
        round_strategy = np.zeros(np.shape(placement_strategy))
        for s in range(self.edge_num):
            for n in range(env.service_num):
                round_strategy[s][n] = self.round_value(placement_strategy[s][n])
        return round_strategy

    def round_offloading_strategy(self, offloading_strategy: List[np.array]) -> np.array:
        round_strategy = np.zeros(np.shape(offloading_strategy))
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                round_strategy[s][m] = self.round_value(offloading_strategy[s][m])
        return round_strategy

    def gen_offloading_strategy(self, x) -> np.array:
        """
        generate offloading strategy by fraction x
        :param x: a 2D dimension matrix
        :return: offloading strategy
        """
        offloading_strategy = np.zeros(self.edge_num, self.mobile_num)
        for m in range(self.mobile_num):
            for s in range(self.edge_num):
                offloading_strategy[s][m] = self.round_value(x[self.get_idx(element="y", s=s, m=m)])
        return offloading_strategy

    def black_measure(self):
        t = 0
        res = self.lr(t)
        model_allocation_space = [0 for _ in range(self.edge_num)]
        edge_computing_capacity = [0 for _ in range(self.edge_num)]
        for s in range(self.edge_num):
            for n in range(self.model_num):
                ratio = res.x[self.get_idx(element="x", s=s, n=n)]
                model_allocation_space[s] += ratio * env.model_size[n]
        print(-res.fun)
        print("using space: ", model_allocation_space)
        print("available space: ", [edge.storage_capacity for edge in self.edge])
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                ratio = res.x[self.get_idx(element="y", s=s, m=m)]
                n = self.mobile[m].task[t][0]
                edge_computing_capacity[s] += env.workload[n] * ratio
        print("using computing capacity: ", edge_computing_capacity)
        print("available computing capacity: ", [edge.computing_capacity for edge in self.edge])
        print(np.sum(res.x[self.y_start: self.y_length + self.y_start] >= 0.5))
        print(res.x[self.y_start: self.y_start + self.y_length])

    def _lr_max_ratio(self, method='interior-point', placement_strategy=None, collaboration=True):
        self.init_idx()
        obj_vector = self.get_obj_vector_max_ratio()
        A_ub, b_ub = self._get_lr_constraint(ratio=True)
        bounds = self.get_bounds()
        res = op.linprog(-obj_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        return res

    def _lr_max_ratio_without(self, method='interior-point', placement_strategy=None, collaboration=False):
        self.init_idx()
        obj_vector = self.get_obj_vector_max_ratio()
        A_ub, b_ub = self._get_lr_constraint(placement_strategy=placement_strategy, collaboration=collaboration, ratio=True)
        bounds = self.get_bounds()
        res = op.linprog(-obj_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        return res

    def _lr_max_throughput(self, method='high', placement_strategy=None, collaboration=True):
        self.init_idx()
        obj_vector = self.get_obj_vector_max_throughput()
        A_ub, b_ub = self._get_lr_constraint(ratio=False)
        bounds = self.get_bounds()
        res = op.linprog(-obj_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        return res

    def lr_max_ratio(self):
        # method = ['highs', 'highs-ds', 'highs-ipm']
        res = self._lr_max_ratio(method='simplex')
        if res.success:
            return res

    def lr_max_ratio_without(self):
        # method = ['highs', 'highs-ds', 'highs-ipm']
        res = self._lr_max_ratio_without(method='simplex')
        if res.success:
            return res

    def lr_max_throughput(self):
        # method = ['highs', 'highs-ds', 'highs-ipm']
        res = self._lr_max_throughput(method='simplex')
        if res.success:
            return res

    def get_obj_vector_max_ratio(self):
        vector = self._get_zero_vector()
        vector[self.get_idx(element="g")] = 1
        return np.array(vector)

    def get_obj_vector_max_throughput(self):
        vector = self._get_zero_vector()
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                vector[self.get_idx(element="y", s=s, m=m)] = 1
        return np.array(vector)

    def _get_zero_vector(self):
        return [0 for _ in range(self.total_length)]

    def _get_lr_constraint(self, ratio=True):
        A_ub, b_ub = [], []
        storage_left, storage_right = self._constraint_storage()
        A_ub.extend(storage_left)
        b_ub.extend(storage_right)

        computing_left, computing_right = self._constraint_computing_capacity()
        A_ub.extend(computing_left)
        b_ub.extend(computing_right)

        execution_one_left, execution_one_right = self._constraint_exection_ones()
        A_ub.extend(execution_one_left)
        b_ub.extend(execution_one_right)

        place_model_left, place_model_right = self._constraint_place_model()
        A_ub.extend(place_model_left)
        b_ub.extend(place_model_right)

        if ratio is True:
            ratio_left, ratio_right = self._constraint_ratio()
            A_ub.extend(ratio_left)
            b_ub.extend(ratio_right)
        return np.array(A_ub), np.array(b_ub)

    def _constraint_storage(self):
        A_ub, b_ub = [], []
        for s in range(self.edge_num):
            vector = self._get_zero_vector()
            for n in range(env.service_num):
                vector[self.get_idx(element="x", s=s, n=n)] = env.service_size[n]
            A_ub.append(vector.copy())
            b_ub.append(self.edge[s].storage_capacity)
        return A_ub, b_ub

    def _constraint_computing_capacity(self):
        A_ub, b_ub = [], []
        for s in range(self.edge_num):
            vector = self._get_zero_vector()
            for m in range(self.mobile_num):
                vector[self.get_idx(element="y", s=s, m=m)] = self.workload[s][m]
            A_ub.append(vector.copy())
            b_ub.append(self.edge[s].computing_capacity)
        return A_ub, b_ub

    def _constraint_exection_ones(self):
        A_ub, b_ub = [], []
        for m in range(self.mobile_num):
            vector = self._get_zero_vector()
            for s in range(self.edge_num):
                vector[self.get_idx(element="y", s=s, m=m)] = 1
            A_ub.append(vector.copy())
            b_ub.append(1)
        return A_ub, b_ub

    def _constraint_place_model(self):
        A_ub, b_ub = [], []
        for m in range(self.mobile_num):
            service_id = self.mobile[m].task_type
            for s in range(self.edge_num):
                vector = self._get_zero_vector()
                vector[self.get_idx(element="x", s=s, n=service_id)] = -1
                vector[self.get_idx(element="y", s=s, m=m)] = 1
                A_ub.append(vector.copy())
                b_ub.append(0)
        return A_ub, b_ub

    def _constraint_ratio(self):
        A_ub, b_ub = [], []
        for n in range(env.service_num):
            if self.task_num[n] == 0:
                continue
            vector = self._get_zero_vector()
            for m in range(self.mobile_num):
                if 1 == self.mobile_service[m][n]:
                    for s in range(self.edge_num):
                        vector[self.get_idx(element="y", s=s, m=m)] = -1 / self.task_num[n]
            vector[self.get_idx(element='g')] = 1
            A_ub.append(vector.copy())
            b_ub.append(0)
        return A_ub, b_ub

    def get_bounds(self):
        bound = [(0, 1) for _ in range(self.total_length)]
        return tuple(bound)

    def get_idx(self, element="x", s=None, m=None, n=None):
        if element == "x":
            return self.x_start + s * env.service_num + n
        if element == "y":
            return self.y_start + s * self.mobile_num + m
        if element == "g":
            return self.g_start

    def round_value(self, gap):
        """
        round a value to 1 with probability gap
        :param gap: probability
        :return:
        """
        v = 0
        if random.random() <= gap:
            v = 1
        return v


class Throughput:
    def __init__(self, mobile: List[env.Mobile], edge: List[env.Edge], environment: env.Environment, workload: np.array) -> None:
        self.mobile = mobile
        self.edge = edge
        self.environment = environment
        self.workload = workload
        self.mobile_service, self.task_num = mobile_belong(self.mobile)
        self.mobile_num = len(self.mobile)
        self.edge_num = len(self.edge)

    def re_init(self, mobile: List[env.Mobile], edge: List[env.Edge], environment: env.Environment, workload):
        self.mobile = mobile
        self.edge = edge
        self.environment = environment
        self.workload = workload
        self.mobile_service, self.task_num = mobile_belong(self.mobile)
        self.mobile_num = len(self.mobile)
        self.edge_num = len(self.edge)

    def service_placement_strategy_check(self, placement_strategy, x):
        # 检测服务缓存是否违背边缘服务器的存储约束，同时尽可能放置更多的服务在边缘服务器中
        for s in range(self.edge_num):
            storage = 0
            for n in range(env.service_num):
                if 1 == placement_strategy[s][n]:
                    storage += env.service_size[n]
            tmp = list(x[s].copy())
            while(storage > self.edge[s].storage_capacity):
                idx = tmp.index(min(tmp))
                if 1 == placement_strategy[s][idx]:
                    placement_strategy[s][idx] = 0
                    storage -= env.service_size[idx]
                tmp[idx] = 2
            tmp = list(x[s].copy())
            for _ in range(env.service_num):
                idx = tmp.index(max(tmp))
                if 0 == placement_strategy[s][idx]:
                    if storage + env.service_size[idx] <= self.edge[s].storage_capacity:
                        placement_strategy[s][idx] = 1
                        storage += env.service_size[idx]
                tmp[idx] = -2
        return placement_strategy

    def service_placement_strategy_check_static(self, placement_strategy, x):
        # 检测服务缓存是否违背边缘服务器的存储约束，同时尽可能放置更多的服务在边缘服务器中
        for s in range(self.edge_num):
            storage = 0
            for n in range(env.service_num):
                if 1 == placement_strategy[s][n]:
                    storage += env.service_size[n]
            tmp = list(x[s].copy())
            while(storage > self.edge[s].storage_capacity):
                idx = tmp.index(min(tmp))
                if 1 == placement_strategy[s][idx]:
                    placement_strategy[s][idx] = 0
                    storage -= env.service_size[idx]
                tmp[idx] = 2
            tmp = list(x[s].copy())
            for _ in range(env.service_num):
                idx = tmp.index(max(tmp))
                if 0 == placement_strategy[s][idx]:
                    if storage + env.service_size[idx] <= self.edge[s].storage_capacity:
                        placement_strategy[s][idx] = 1
                        storage += env.service_size[idx]
                tmp[idx] = -2
        return placement_strategy

    def check_placement_strategy(self, placement_strategy) -> bool:
        # # 检测服务缓存是否违背边缘服务器的存储约束
        for s in range(self.edge_num):
            storage = 0
            for n in range(env.service_num):
                if 1 == placement_strategy[s][n]:
                    storage += env.service_size[n]
            if storage > self.edge[s].storage_capacity:
                print("存储约束检测失败")
                return False
        return True

    def check_offloading_strategy(self, placement_strategy, offloading_strategy) -> bool:
        for m in range(self.mobile_num):
            a = 0
            for s in range(self.edge_num):
                if 1 == offloading_strategy[s][m]:
                    a += 1
                    if a > 1:
                        print("1个用户分配到多个服务器")
                        return False
        for s in range(self.edge_num):
            compute = 0
            for m in range(self.mobile_num):
                if 1 == offloading_strategy[s][m]:
                    service_id = self.mobile[m].task_type
                    if placement_strategy[s][service_id] == 0:
                        print("任务调度到了没有缓存服务的边缘服务器")
                        return False
                    compute += self.workload[s][m]
            if compute > self.edge[s].computing_capacity:
                print("计算约束检测失败")
                return False
        return True

    def modify_offloading_by_correct_edge(self, service_placement_strategy, offloading_strategy) -> np.array:
        for m in range(self.mobile_num):
            service_id = self.mobile[m].task_type
            for s in range(self.edge_num):
                if offloading_strategy[s][m] == 1 and service_placement_strategy[s][service_id] == 0:
                    # edge does not place the corresponding model
                    offloading_strategy[s][m] = 0
        return offloading_strategy

    def offloading_strategy_non_overlapping(self, offloading_strategy, y):
        offloading_count = np.sum(offloading_strategy, axis=0)
        for m in range(self.mobile_num):
            if offloading_count[m] > 1:
                tmp = []
                for s in range(self.edge_num):
                    tmp.append(y[s][m])
                while(offloading_count[m] > 1):
                    idx = tmp.index(min(tmp))
                    if 1 == offloading_strategy[idx][m]:
                        offloading_strategy[idx][m] = 0
                        offloading_count[m] -= 1
                    tmp[idx] = 2
        return offloading_strategy

    def offloading_strategy_check_computing_constraints(self, placement_strategy, offloading_strategy, y):
        for s in range(self.edge_num):
            compute = 0
            for m in range(self.mobile_num):
                if 1 == offloading_strategy[s][m]:
                    compute += self.workload[s][m]
            tmp = list(y[s].copy())
            while(compute > self.edge[s].computing_capacity):
                # print(compute)
                idx = tmp.index(min(tmp))
                if offloading_strategy[s][idx] == 1:
                    offloading_strategy[s][idx] = 0
                    compute -= self.workload[s][idx]
                tmp[idx] = 2
            tmp = list(y[s].copy())
            for _ in range(self.mobile_num):
                idx = tmp.index(max(tmp))
                sid = self.mobile[idx].task_type
                if 1 == placement_strategy[s][sid]:
                    if compute + self.workload[s][idx] <= self.edge[s].computing_capacity:
                        offloading_strategy[s][idx] = 1
                        compute += self.workload[s][idx]
                tmp[idx] = -2
        return offloading_strategy

    def offloading_strategy_max_ratio(self, service_placement_strategy, offloading_strategy):
        compute_load = []
        offloading_count = np.sum(offloading_strategy, axis=0)
        for _ in range(self.edge_num):
            compute_load.append(0)
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                if offloading_strategy[s][m] == 1:
                    compute_load[s] += self.workload[s][m]
        flag = 0
        while 1 > 0:
            flag = 0
            _, n = self.generate_ratio(offloading_strategy)
            user = []
            for m in range(self.mobile_num):
                if self.mobile_service[m][n] == 1 and offloading_count[m] == 0:
                    user.append(m)
            minLoad = 100000
            ids = -1
            idm = -1
            for i in range(len(user)):
                m = user[i]
                for s in range(self.edge_num):
                    if compute_load[s] + self.workload[s][m] <= self.edge[s].computing_capacity:
                        if service_placement_strategy[s][n] == 1:
                            if self.workload[s][m] < minLoad:
                                minLoad = self.workload[s][m]
                                ids = s
                                idm = m
                                flag = 1
            if flag == 0:
                break
            offloading_strategy[ids][idm] = 1
            compute_load[ids] += self.workload[ids][idm]
            offloading_count[idm] = 1
        return offloading_strategy

    def offloading_strategy_max_throughput(self, service_placement_strategy, offloading_strategy):
        compute_load = []
        offloading_count = np.sum(offloading_strategy, axis=0)
        for _ in range(self.edge_num):
            compute_load.append(0)
        for s in range(self.edge_num):
            for m in range(self.mobile_num):
                if offloading_strategy[s][m] == 1:
                    compute_load[s] += self.workload[s][m]
        flag = 0
        while 1 > 0:
            flag = 0
            for m in range(self.mobile_num):
                if offloading_count[m] == 0:
                    n = self.mobile[m].task_type
                    for s in range(self.edge_num):
                        if service_placement_strategy[s][n] == 1:
                            if compute_load[s] + self.workload[s][m] <= self.edge[s].computing_capacity:
                                offloading_strategy[s][m] = 1
                                compute_load[s] += self.workload[s][m]
                                offloading_count[m] = 1
                                flag = 1
                                break
            if flag == 0:
                break
        return offloading_strategy

    def generate_ratio(self, offloading_strategy: np.array):
        ratio = [0 for _ in range(env.service_num)]
        ans = []
        offloading_count = np.sum(offloading_strategy, axis=0)
        for s in range(env.service_num):
            for m in range(self.mobile_num):
                if 1 == self.mobile_service[m][s]:
                    if offloading_count[m] == 1:
                        ratio[s] += 1
        for n in range(env.service_num):
            if self.task_num[n] != 0:
                ans.append(ratio[n] / self.task_num[n])
            else:
                ans.append(1)
        return min(ans), ans.index(min(ans))

    def generate_ratio_random(self, offloading_strategy: np.array):
        ratio = [0 for _ in range(env.service_num)]
        ans = []
        offloading_count = np.sum(offloading_strategy, axis=0)
        for s in range(env.service_num):
            for m in range(self.mobile_num):
                if 1 == self.mobile_service[m][s]:
                    if offloading_count[m] == 1:
                        ratio[s] += 1
        for n in range(env.service_num):
            if self.task_num[n] != 0:
                ans.append(ratio[n] / self.task_num[n])
            else:
                ans.append(1)
        minRatio = min(ans)
        minSevice = []
        for i in range(env.service_num):
            if minRatio == ans[i]:
                minSevice.append(i)
        return minSevice[random.randint(0, len(minSevice)-1)]

    def generate_ratio_print(self, offloading_strategy: np.array):
        """
        this function should be used after self.set_collaboration
        :param t:
        :param model_placement_strategy:
        :param offloading_strategy:
        :return: throughput -> int
        """
        ratio = [0 for _ in range(env.service_num)]
        ans = []
        offloading_count = np.sum(offloading_strategy, axis=0)
        for s in range(env.service_num):
            for m in range(self.mobile_num):
                if 1 == self.mobile_service[m][s]:
                    if offloading_count[m] == 1:
                        ratio[s] += 1
        for n in range(env.service_num):
            if self.task_num[n] != 0:
                ans.append(round(ratio[n] / self.task_num[n], 2))
            else:
                ans.append(1)
        # print(ans)

    # def modify_offloading_by_computing_capacity(self, model_placement_strategy, offloading_strategy):
    #     computing_capacity = [0 for _ in range(self.edge_num)]
    #     for s in range(self.edge_num):
    #         for m in range(self.mobile_num):
    #             model_idx = self.mobile[m].chose_model
    #             if offloading_strategy[s][m] == 1 and model_placement_strategy[s][model_idx] == 1:
    #                 computing_capacity[s] += self.workload[s][m]
    #         if computing_capacity[s] > self.edge[s].computing_capacity:
    #             a = []
    #             for m in range(self.mobile_num):
    #                 model_idx = self.mobile[m].chose_model
    #                 if offloading_strategy[s][m] == 1 and model_placement_strategy[s][model_idx] == 1:
    #                     a.append(m)
    #             for _m in a:
    #                 model_idx = self.mobile[_m].chose_model
    #                 computing_capacity[s] -= self.workload[s][m]
    #                 offloading_strategy[s][_m] = 0
    #                 if computing_capacity[s] <= self.edge[s].computing_capacity:
    #                     break
    #     return offloading_strategy

    # def get_computing_capacity(self, offloading_strategy, workload):
    #     edge_capacity = np.zeros(self.edge_num)
    #     for s in range(self.edge_num):
    #         for m in range(self.mobile_num):
    #             if offloading_strategy[s][m] == 1:
    #                 edge_capacity[s] += workload[s][m]
    #     return edge_capacity
