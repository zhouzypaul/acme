import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph (example edges)
G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_edge('C', 'D')
G.add_edge('D', 'C')  # Cycle example
G.add_edge('E', 'F')
G.add_edge('F', 'C')
G.add_edge('G', 'G')
G.add_edge('G', 'A')

# Goal node
goal_node = 'C'

node2path = nx.single_target_shortest_path(G, target='C')

for node, path in node2path.items():
  print(f'Path from {node} to {goal_node}: {path[1:]}')

for node, path in node2path.items():
  action = path[1] if len(path) > 1 else path[0]
  print(f'Action from {node} -> {action}')
