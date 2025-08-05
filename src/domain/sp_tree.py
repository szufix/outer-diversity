from itertools import product

from src.domain.validator import validate_domain
from src.domain.extenders import extend_to_reversed

def generate_single_peaked_votes(tree_adj):
    # Ensure the tree is undirected
    for node, neighbors in list(tree_adj.items()):
        for neighbor in neighbors:
            if node not in tree_adj[neighbor]:
                tree_adj[neighbor].append(node)

    def build_tree(u, parent):
        children = []
        for v in tree_adj[u]:
            if v != parent:
                children.append(build_tree(v, u))
        return (u, children)

    def interleave(seqs):
        # Recursively yield all interleavings of multiple sequences
        if not any(seqs):
            yield []
            return
        for i, seq in enumerate(seqs):
            if seq:
                for rest in interleave(seqs[:i] + [seq[1:]] + seqs[i+1:]):
                    yield [seq[0]] + rest

    def sp_orders(tree):
        root, subtrees = tree
        if not subtrees:
            return [[root]]

        sub_orders_lists = [sp_orders(sub) for sub in subtrees]
        all_combinations = product(*sub_orders_lists)
        results = []
        for combo in all_combinations:
            for inter in interleave(list(combo)):
                results.append([root] + inter)
        return results

    all_votes = set()
    for peak in tree_adj:
        rooted = build_tree(peak, None)
        for vote in sp_orders(rooted):
            all_votes.add(tuple(vote))

    return [list(v) for v in all_votes]

############################################################

def create_path_tree(n):
    tree = {i: [] for i in range(n)}
    for i in range(n - 1):
        tree[i].append(i + 1)
        tree[i + 1].append(i)
    return tree

def create_star_tree(n):
    if n < 2:
        raise ValueError("Star tree must have at least 2 nodes.")
    tree = {i: [] for i in range(n)}
    center = 0
    for i in range(1, n):
        tree[center].append(i)
        tree[i].append(center)
    return tree

def create_double_forked_tree(length):
    length -= 4
    if length < 1:
        raise ValueError("Number of candidates must be at least 5.")

    tree = {}
    node_id = 0

    # Create path nodes N0 to Nk
    path_nodes = []
    for _ in range(length):
        tree[node_id] = []
        path_nodes.append(node_id)
        node_id += 1

    # Connect path as a line
    for i in range(length - 1):
        tree[path_nodes[i]].append(path_nodes[i + 1])
        tree[path_nodes[i + 1]].append(path_nodes[i])

    # Add two leaves at the start
    for _ in range(2):
        tree[node_id] = [path_nodes[0]]
        tree[path_nodes[0]].append(node_id)
        node_id += 1

    # Add two leaves at the end
    for _ in range(2):
        tree[node_id] = [path_nodes[-1]]
        tree[path_nodes[-1]].append(node_id)
        node_id += 1

    return tree

def create_triple_forked_tree(length):
    length -= 6
    if length < 1:
        raise ValueError("Length must be at least 7.")

    tree = {}
    node_id = 0

    # Create path nodes
    path_nodes = []
    for _ in range(length):
        tree[node_id] = []
        path_nodes.append(node_id)
        node_id += 1

    # Connect central path
    for i in range(length - 1):
        tree[path_nodes[i]].append(path_nodes[i + 1])
        tree[path_nodes[i + 1]].append(path_nodes[i])

    # Add 3 leaves to the start of the path (N0)
    for _ in range(3):
        tree[node_id] = [path_nodes[0]]
        tree[path_nodes[0]].append(node_id)
        node_id += 1

    # Add 3 leaves to the end of the path (Nk)
    for _ in range(3):
        tree[node_id] = [path_nodes[-1]]
        tree[path_nodes[-1]].append(node_id)
        node_id += 1

    return tree

############################################################

@validate_domain
def sp_path_domain(num_candidates):
    tree = create_path_tree(num_candidates)
    return generate_single_peaked_votes(tree)

@validate_domain
def sp_star_domain(num_candidates):
    tree = create_star_tree(num_candidates)
    return generate_single_peaked_votes(tree)

@validate_domain
def sp_double_forked_domain(num_candidates):
    tree = create_double_forked_tree(num_candidates)
    return generate_single_peaked_votes(tree)

@validate_domain
def sp_triple_forked_domain(num_candidates):
    tree = create_triple_forked_tree(num_candidates)
    return generate_single_peaked_votes(tree)

@validate_domain
def ext_sp_double_forked_domain(num_candidates):
    domain = sp_double_forked_domain(num_candidates)
    return extend_to_reversed(domain)