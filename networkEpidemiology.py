import networkx as nx
import numpy as np
from collections import defaultdict

def simulate_sir(graph, beta=0.3, gamma=0.1, initial_infecteds=1, steps=200):
    status = {node: 'S' for node in graph.nodes()}
    infecteds = np.random.choice(list(graph.nodes()), initial_infecteds)
    for node in infecteds:
        status[node] = 'I'
    
    history = defaultdict(list)
    for state in ['S', 'I', 'R']:
        history[state].append(sum(1 for v in status.values() if v == state))
    
    for t in range(steps):
        new_status = status.copy()
        for node in graph.nodes():
            if status[node] == 'I':
                if np.random.random() < gamma:
                    new_status[node] = 'R'
                for neighbor in graph.neighbors(node):
                    if status[neighbor] == 'S' and np.random.random() < beta:
                        new_status[neighbor] = 'I'
        status = new_status
        for state in ['S', 'I', 'R']:
            history[state].append(sum(1 for v in status.values() if v == state))
    
    return history

G = nx.barabasi_albert_graph(1000, 5)
history = simulate_sir(G)