import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations, permutations
from numba import jit, prange
from tqdm import tqdm
from numba.typed import List
from numba import types
import random
from tqdm import tqdm

class GraphVisualization:
    def __init__(self, graph, title, additional_edge = False):
        self.visual = []
        self.graph = graph
        self.additional_edge = additional_edge
        self.title = title
        
    def addEdge(self, *a):
        temp = (a)
        self.visual.append(temp)

    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.title(self.title)
        # plt.show()
        

    def visualize_full_graph(self):
        for i in range(len(self.graph)):
            for v in self.graph[i]:
                self.addEdge(i, v)
        self.visualize()

def all_independent_sets(adj, k_max=None, oriented = True):

    n = len(adj)
    if k_max is None:
        k_max = int(n /2) +1

    if oriented == True:                                       
        is_edge = lambda u, v: v in adj[u] or u in adj[v]
    else:
        is_edge = lambda u, v: v in adj[u] 

    independents = List()

    for size in range(0, k_max + 1):
        for subset in combinations(range(n), size):
            if len(subset) == 0:
                pass
            else:
                if all(not is_edge(u, v) for u, v in combinations(subset, 2)):
                    independents.append(List(list(subset)))

    return independents

def visualize_graph(graph, title):
    G = GraphVisualization(graph, title)
    G.visualize_full_graph()

def graph_data(graph, title):
    G = GraphVisualization(graph, title)
    G.visualize_full_graph()
    print("Adjacency list", graph)
    independents = all_independent_sets(graph)
    permutation_sets = set_permutation_sets(independents)

    print("Number of independent sets = ", len(independents))
    print("Number of permutations = ", len(permutation_sets))
    print("Set of independent sets", independents)
    print("set of permutations", permutation_sets)
    return independents, permutation_sets

def compute_permutation_set(graph):
    independents = all_independent_sets(graph)
    permutation_sets = set_permutation_sets(independents)
    return independents, permutation_sets

def set_permutation_sets(independent_sets):
    permutation_list = List()

    for values in independent_sets:
        for permut in permutations(values):
            inner = List()
            if len(permut) !=0:
                for v in permut:
                    inner.append(v)
            permutation_list.append(inner)
    return permutation_list

@jit(nopython = True)
def alpha(graph):
    alpha_value = List()
    for i in range(len(graph)):
        alpha_value.append(1/len(graph))
    return alpha_value

@jit(nopython = True)
def neighbour(given_set, graph):
    neighbour_set = List.empty_list(types.int64)
    for s in given_set:
        for g in graph[s]:
            if g in neighbour_set:
                pass
            else:
                neighbour_set.append(g)
    return neighbour_set


@jit(nopython = True)
def card_alpha_set(given_set, alpha_value):
    sum_alpha = 0.0
    for s in given_set:
        sum_alpha += alpha_value[s]
    return sum_alpha

@jit(nopython = True)
def T_I_rond(given_set, alpha_value, graph):
    product = 1
    for k in prange(len(given_set)):
        cardinal_neighbour = card_alpha_set(neighbour(given_set[:(k+1)], graph), alpha_value) 
        cardinal_set = card_alpha_set(given_set[:(k+1)], alpha_value)
        product *= alpha_value[given_set[k]] / (cardinal_neighbour - cardinal_set)
    return product

@jit(nopython = True, parallel = True)
def T_I(set_permutation_sets, graph, alpha_value, independent_set):
    TI_value = 0.
    if is_stable(graph, alpha_value, independent_set) == False:
        return np.nan
    for key in prange(len(set_permutation_sets)):
        TI_value += T_I_rond(set_permutation_sets[key], alpha_value, graph)
    return TI_value
@jit(nopython = True)
def E_I_rond(given_set, alpha_value, graph):
    left_sum = 0
    for l in prange(len(given_set)):
        cardinal_neighbour = card_alpha_set(neighbour(given_set[:(l+1)], graph), alpha_value)
        cardinal_set =  card_alpha_set(given_set[:(l+1)], alpha_value)  
        left_sum += cardinal_neighbour / (cardinal_neighbour - cardinal_set)

    right_product = T_I_rond(given_set, alpha_value, graph)
    return left_sum*right_product

@jit(nopython = True)
def is_stable(graph, alpha_value, independent):
    is_higher = False
    for node in independent:
        
        alpha_neighbor_graph, alpha_graph = 0., 0.
        for i in neighbour(node, graph):
            alpha_neighbor_graph += alpha_value[i]
        for i in node:
            alpha_graph += alpha_value[i]
        if alpha_graph >= alpha_neighbor_graph - 1e-8:
            is_higher = True

    if is_higher == True:
        return False
    else:
        return True
        
@jit(nopython = True, parallel = True)
def E_I(set_permutation_sets, graph, alpha_value, independent_set):
    if is_stable(graph, alpha_value, independent_set) == False:
        return np.nan
    EI_value = 0.
    for key in prange(len(set_permutation_sets)):
        EI_value += E_I_rond(set_permutation_sets[key], alpha_value, graph)
    return EI_value

def expected_value_calculation(e_i, t_i):
    return e_i/(t_i+1)

@jit(nopython = True)
def expected_value_given_alpha(alpha_value, permutation, graph, independent):
    EI_val = E_I(permutation, graph, alpha_value, independent)
    TI_val = T_I(permutation, graph, alpha_value, independent)
    return EI_val / (1 + TI_val)

def size_number_nodes(graph):
    alpha_test = alpha(graph)
    independents = all_independent_sets(graph)
    permutation_sets = set_permutation_sets(independents)

    TI_val = T_I(permutation_sets, graph, alpha_test, independents)
    EI_val = E_I(permutation_sets, graph, alpha_test, independents)

    expected_value = expected_value_calculation(EI_val, TI_val)
    return expected_value

def compute_expectation_single_size(graph_list, graph_name, additional_cycle, title):
    plt.figure()
    expected_value_list = []
    for graph, name in zip(graph_list, graph_name):
        expected_value = size_number_nodes(graph)
        expected_value_list.append(expected_value)
    plt.scatter(additional_cycle, expected_value_list)
    plt.xlabel("Number of nodes in the inner cycle")
    plt.ylabel("Expected value")
    plt.title(title)

def pipeline_expected_value(min_nodes, max_nodes, triangle):
    x = np.arange(min_nodes, max_nodes+1, 2)
    tab_expected_value = np.zeros((len(x)))
    tab_independent_sets = np.zeros((len(x)))
    tab_permutation_sets = np.zeros((len(x)))

    for index, node in enumerate(x):
        graph = generate_cycle_graph(node, triangle)
        independents = all_independent_sets(graph)
        permutation_sets = set_permutation_sets(independents)
        
        alpha_value = alpha(graph)
        EI_value = E_I(permutation_sets, graph, alpha_value, independents)
        TI_value = T_I(permutation_sets, graph, alpha_value, independents)

        expected_value = expected_value_calculation(EI_value, TI_value)


        tab_expected_value[index] = expected_value
        tab_independent_sets[index] = len(independents)
        tab_permutation_sets[index] =  len(permutation_sets)
    return x, tab_expected_value, tab_independent_sets, tab_permutation_sets

@jit(nopython = True)
def sample_from_probs(alpha):
    r = np.random.rand()
    cum = 0.0
    for i in range(len(alpha)):
        cum += alpha[i]
        if r < cum:
            return i
    return len(alpha) - 1

@jit(nopython = True)
def compute_expectation_monte_carlo(graph, alpha_value, max_iter):
    item_list = List()
    number_elements = List()
    variance = 0.0
    for i in range(max_iter):
        item = sample_from_probs(alpha_value)        
        break_loop = False
        for idx, item_l in enumerate(item_list):
            for node in graph[item]:
                if node == item_l:
                    del item_list[idx]
                    break_loop = True
                    break
            if break_loop:
                break

        if not break_loop and i >= 0.03*max_iter:
            item_list.append(item)

        if i >= 0.03*max_iter:
            number_elements.append(len(item_list))


    esperance = sum(number_elements)/len(number_elements)
    for element in number_elements:
        variance += (element - esperance)**2
    variance = variance/len(number_elements)

    return esperance, (variance/len(number_elements))**(0.5)

def monte_carlo_comparison(graph_list, graph_name, max_iter):
    for i in range(len(graph_list)):
        alpha_value = alpha(graph_list[i])
        alpha_value = np.array(alpha_value / np.sum(alpha_value))
        independent, permutation = compute_permutation_set(graph_list[i])
        T_I_value = T_I(permutation, graph_list[i], alpha_value, independent)
        E_I_value = E_I(permutation, graph_list[i], alpha_value, independent)
        experance_analytique = expected_value_calculation(E_I_value, T_I_value)
        esperance_mc, std = compute_expectation_monte_carlo(graph_list[i], alpha_value, max_iter)
        print(f"Graph {graph_name[i]} : Analytical formula expected value = {experance_analytique:.4f} | Expected value Monte Carlo = {esperance_mc:.4f} | std Monte Carlo = {std:.5f}")

    
def generate_cycle_graph(n, additional_edge):
    graph = List()
    for i in range(n):
        
        if i ==0:
            graph.append(List([np.int64(n-1), np.int64(i+1)]))
        elif i == n-1:
            graph.append(List([np.int64(i-1), np.int64(0)]))
        else:
            graph.append(List([np.int64(i-1), np.int64(i+1)]))
    if additional_edge == 0:
        return graph
    if additional_edge !=0:
        graph[0].append(additional_edge-1) 
        graph[additional_edge-1].append(0) 
    return graph
@jit(nopython = True)
def generate_alpha(graph):
    list_alpha = List()
    for i in range(len(graph)):
        list_alpha.append(random.uniform(0, 1))
    sum_list = sum(list_alpha)
    for i in range(len(list_alpha)):
        list_alpha[i] = list_alpha[i]/sum_list
    return list_alpha

def expected_value_computation(graph, alpha_value, max_iter, graph_name):
    for i in range(len(graph)):
        independent, permutation = compute_permutation_set(graph[i])
        expected_value_graph = expected_value_given_alpha(alpha_value, permutation, graph[i], independent)
        expectation_mc, std_mc = compute_expectation_monte_carlo(graph[i], alpha_value, max_iter)
        print("Analytical expected value", str(graph_name[i]), round(expected_value_graph, 3), "Monte Carlo", str(graph_name[i]), round(expectation_mc, 3), "std Monte Carlo", round(std_mc, 6))


@jit(nopython = True)
def find_paradox(n_trials, graph, graph_modif, permutation_normal, permutation_modif, independent_normal, independent_modif, print_mc, graph_name, additional_edge):

    selected_alpha = List()
    selected_expected_values = List()
    for i in prange(n_trials):
        alpha_value = generate_alpha(graph)
        if (is_stable(graph, alpha_value, independent_normal) == True) :
            if (is_stable(graph_modif, alpha_value, independent_modif) == True):
                expected_value_graph = expected_value_given_alpha(alpha_value, permutation_normal, graph, independent_normal)
                expected_value_modif = expected_value_given_alpha(alpha_value, permutation_modif, graph_modif, independent_modif)
                if expected_value_modif > expected_value_graph:
                    selected_alpha.append(alpha_value)
                    selected_expected_values.append(List([expected_value_modif, expected_value_graph]))
                    if print_mc:
                        expected_val_normal, std_normal = compute_expectation_monte_carlo(graph, alpha_value, int(1e7))
                        expected_val_modif, std_modif = compute_expectation_monte_carlo(graph_modif, alpha_value, int(1e7))
                        print("Analytical expected value", str(graph_name), round(expected_value_graph, 3), "Monte Carlo", str(graph_name), round(expected_val_normal, 3), "std Monte Carlo", round(std_normal, 6), "Analytical expected value", str(graph_name), "graph +", str(additional_edge),  round(expected_value_modif, 3), "Monte Carlo", round(expected_val_modif, 3), "std Monte Carlo", round(std_modif, 6))
                    else:
                        print("Alpha value", alpha_value, "Analytical expected value", graph_name, round(expected_value_graph, 3), "Analytical expected value", graph_name, "graph +", additional_edge, round(expected_value_modif, 3))
            else:
                pass
        else:
            pass
    alpha_value = alpha(graph)
    expected_value_graph = expected_value_given_alpha(alpha_value, permutation_normal, graph, independent_normal)
    expected_value_modif = expected_value_given_alpha(alpha_value, permutation_modif, graph_modif, independent_modif)
    return selected_alpha, selected_expected_values

