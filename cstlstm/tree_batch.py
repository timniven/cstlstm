"""Tree data structures and functions for parallel processing."""
import numpy as np
from nltk.tokenize import sexpr


def cumsum(seq):
    """Get the cumulative sum of a sequence of sequences at each index.

    Args:
      seq: List of sequences.

    Returns:
      List of integers.
    """
    r, s = [], 0
    for e in seq:
        l = len(e)
        r.append(l + s)
        s += l
    return r


def flatten_list_of_lists(list_of_lists):
    """Flatten a list of lists.

    Args:
      list_of_lists: List of Lists.

    Returns:
      List.
    """
    return [item for sub_list in list_of_lists for item in sub_list]


def get_adj_mat(nodes):
    """Get an adjacency matrix from a node set.

    A row in the matrix indicates the children of the node at that index.
    A column in the matrix indicates the parent of the node at that index.

    Args:
      nodes: List of Nodes.

    Returns:
      2D numpy.ndarray: an adjacency matrix.
    """
    size = len(nodes)
    mat = np.zeros((size, size), dtype='int32')
    for node in nodes:
        if node.parent_id >= 0:
            mat[node.parent_id][node.id] = 1
    return mat


def get_child_ixs(nodes, adj_mat):
    """Get lists of children indices at each level.

    We need this for batching, to show the wiring of the nodes at each level,
    as we process them in parallel.

    Args:
      nodes: Dictionary of {Integer: [List of Nodes]} for the nodes at each
        level in the tree / forest.
      adj_mat: 2D numpy.ndarray, adjacency matrix for all nodes.

    Returns:
      Dictionary of {Integer: [[List of child_ixs @ l+1] for parent_ixs @ l]}.
    """
    child_ixs = {}
    # We don't need child_ixs for the last level so just range(max_level) not +1
    for l in range(max(nodes.keys())):
        child_nodes = nodes[l+1]
        id_to_ix = {child_nodes[ix].id: ix for ix in
                    range(len(child_nodes))}
        ids = [np.nonzero(adj_mat[n.id])[0] for n in nodes[l]]
        try:
            ixs = [[id_to_ix[id] for id in id_list] for id_list in ids]
        except Exception as e:
            print('level: %s' % l)
            print('child_ixs state')
            print(child_ixs)
            print('child_nodes')
            print(child_nodes)
            print('id_to_ix')
            print(id_to_ix)
            raise e
        child_ixs[l] = ixs
    return child_ixs


def get_max_level(nodes):
    """Get the highest level number given a list of nodes.

    Args:
      nodes: List of Nodes.

    Returns:
      Integer, the highest level number. It is a zero-based number, so if later
        the actual number of levels is desired, will need to add one to this.
    """
    return max([n.level for n in nodes])


def get_nodes_at_levels(nodes):
    """Get a dictionary listing nodes at each level.

    Args:
      nodes: List of Nodes.

    Returns:
      Dictionary of {Integer: [List of Nodes]} for each level.
    """
    max_level = get_max_level(nodes)
    return dict(zip(
        range(max_level+1),
        [[n for n in nodes if n.level == l]
         for l in range(max_level+1)]))


def get_parent_ixs(nodes, adj_mat):
    """Get lists of parent indices at each level.

    We need this for batching, to show the wiring of the nodes at each level,
    as we process them in parallel.

    Args:
      nodes: Dictionary of {Integer: [List of Nodes]} for the nodes at each
        level in the tree / forest.
      adj_mat: 2D numpy.ndarray, adjacency matrix for all nodes.

    Returns:
      Dictionary of {Integer: [List of parent_ixs @ l-1 for child_ixs @ l]}.
    """
    parent_ixs = {}
    # We don't need parent_ixs for the first level, 0.
    for l in range(1, max(nodes.keys()) + 1):
        parent_nodes = nodes[l - 1]
        id_to_ix = {parent_nodes[ix].id: ix for ix in
                    range(len(parent_nodes))}
        ids = [np.nonzero(adj_mat[:, n.id])[0][0] for n in nodes[l]]
        ixs = [id_to_ix[id] for id in ids]
        parent_ixs[l] = ixs
    return parent_ixs


def offset_node_lists(node_lists):
    """Offset the ids in the list of node lists.

    Args:
      node_lists: List of Lists of Nodes.

    Returns:
      List of Lists of Nodes.
    """
    cumsums = cumsum(node_lists)
    for list_ix in range(len(node_lists)):
        for node in node_lists[list_ix]:
            offset = cumsums[list_ix - 1] if list_ix > 0 else 0
            node.id = node.id + offset
            node.parent_id = node.parent_id + offset \
                if node.parent_id > 0 \
                else -1
            node.text_ix = node.text_ix + offset
    return node_lists


# Model Classes


class Forest:
    """Forest data structure.

    Designed for the parallel processing of trees in a batch. Will offset ixs
    of it's constituent trees, and define global wirings between all levels,
    allowing each level to be processed in parallel either upwards or downwards.

    Attributes:
      trees: List of Trees.
      node_list: List of all nodes in the forest.
      nodes: Dictionary of {Integer: [List of Nodes]}, defining the nodes at
        each level of depth.
      size: Integer, the number of nodes in the forest.
      max_level: Integer, the maximum level (depth) of the deepest tree in the
        forest.
      adj_mat: 2d numpy.array, adjacency matrix for all nodes.
      child_ixs: Dictionary {Int: [List of List of ixs]}, defining the upward
        wirings.
      parent_ixs: Dictionary {Int: [List of ixs]}, defining the downward
        wirings.
    """

    def __init__(self, trees):
        """Create a new Forest.

        Args:
          trees: List of Trees. They will be processed in order. Pass them in
            the desired order.
        """
        self.trees = trees
        node_lists = offset_node_lists([tree.node_list for tree in trees])
        self.node_list = flatten_list_of_lists(node_lists)
        self.nodes = get_nodes_at_levels(self.node_list)
        self.size = len(self.node_list)
        self.max_level = get_max_level(self.node_list)
        self.adj_mat = get_adj_mat(self.node_list)
        self.child_ixs = get_child_ixs(self.nodes, self.adj_mat)
        #self.parent_ixs = get_parent_ixs(self.nodes, self.adj_mat)


class Node:
    """Node data structure.

    Attributes:
      tag: String, the tag of the token - e.g. VBP.
      pos: String, the part of speech - e.g. VERB.
      token: String, the text of the token - e.g. 'do'.
      id: Integer, the unique id of the node in it's original tree.
      parent_id: Integer, the unique id of the node's prent in it's original
        tree. For ROOT nodes, this should be -1 by convention.
      relationship: String, the relation of this node to the parent - e.g.
        'aux'. For the ROOT of a tree, this should be 'ROOT' by convention.
      text_ix: Integer, the unique index of this node in the order of text, if
        any.
      level: Integer, the level (depth) this node is on in it's original tree,
        where the ROOT level is zero-indexed.
      is_leaf: Boolean indicating whether this node is a leaf.
    """

    def __init__(self, tag, pos, token, id, parent_id, relationship, text_ix,
                 level, is_leaf):
        """Create a new Node."""
        self.tag = tag
        self.pos = pos
        self.token = token
        self.id = id
        self.parent_id = parent_id
        self.relationship = relationship
        self.text_ix = text_ix
        self.level = level
        self.is_leaf = is_leaf
        self.has_token = token is not None
        self.vocab_ix = None  # For vocab_dict index

    def __repr__(self):
        return '\n'.join(['%s: %s' % (key, value)
                          for key, value
                          in self.__dict__.items()])


class Tree:
    """Tree data structure.

    Attributes:
      node_list: List of all nodes in the tree.
      nodes: Dictionary {Int: [List of Nodes]}, giving the Nodes at each level.
      size: Integer, the count of the nodes in the tree.
      max_level: Integer, the max level (depth) of the tree.
      adj_mat: 2D numpy.ndarray, an adjacency matrix giving the relationships
        between all nodes.
      child_ixs: Dictionary {Int: [List of List of ixs]}, defining the upward
        wirings.
      parent_ixs: Dictionary {Int: [List of ixs]}, defining the downward
        wirings.
    """

    def __init__(self, nodes):
        """Create a new Tree.

        Args:
          nodes: List of Nodes.
        """
        self.node_list = nodes
        self.nodes = get_nodes_at_levels(self.node_list)
        self.size = len(self.node_list)
        self.max_level = get_max_level(self.node_list)
        self.adj_mat = get_adj_mat(self.node_list)
        self.child_ixs = get_child_ixs(self.nodes, self.adj_mat)
        #self.parent_ixs = get_parent_ixs(self.nodes, self.adj_mat)


# Parsing Classes and Functions


class Queue:
    def __init__(self):
        self.data = []

    def empty(self):
        return len(self.data) == 0

    def push(self, token, level):
        self.data.append((token, level))

    def pop(self):
        token, level = self.data[0]
        del self.data[0]
        return token, level


class Stack:
    def __init__(self):
        self.items = []

    def empty(self):
        return len(self.items) == 0

    def push(self, sexpr, level, parent_ix):
        self.items.append((sexpr, level, parent_ix))

    def pop(self):
        sexpr, level, parent_ix = self.items[-1]
        del self.items[-1]
        return sexpr, level, parent_ix


# Parsing SpaCy Sents


def sent_to_tree(sent):
    nodes = []
    q = Queue()
    head = next(t for t in sent if t.head == t)
    q.push(head, 0)
    while not q.empty():
        token, level = q.pop()
        node = token_to_node(token, level)
        nodes.append(node)
        for child in token.children:
            q.push(child, level + 1)
    return Tree(nodes)


def token_to_node(token, level):
    return Node(
        tag=token.tag_,
        pos=token.pos_,
        token=token.text,
        id=token.i,
        parent_id=token.head.i if token.head.i != token.i else -1,
        relationship=token.dep_,
        text_ix=token.i,
        level=level,
        is_leaf=len(list(token.children)) == 0)


# Parsing S-Expressions


def tokenize(x):
    """Tokenizes S-expression dependency parse trees that come with NLI data.

    This one has been tested here:
    https://github.com/timniven/hsnli/blob/master/hsnli/tests/tree_sexpr_tests.py

    Args:
      x: String, the tree (or subtree) S-expression.

    Returns:
      String, List(String), Boolean: tag, [S-expression for the node], is_leaf
        flag indicating whether this node is a leaf.
    """
    remove_outer_brackets = x[1:-1]
    if '(' not in remove_outer_brackets:  # means it's a leaf
        split = remove_outer_brackets.split(' ')
        tag, data = split[0], [split[1]]
    else:
        sexpr_tokenized = sexpr.sexpr_tokenize(remove_outer_brackets)
        tag = sexpr_tokenized[0]
        del sexpr_tokenized[0]
        data = sexpr_tokenized
    is_leaf = len(data) == 1 and not (data[0][0] == '(' and data[0][-1] == ')')
    return tag, data, is_leaf


def sexpr_to_tree(sexpr):
    """Returns all nodes in a tree.

    Args:
      sexpr: String, a sexpr.

    Returns:
      Tree.
    """
    nodes = []
    id = -1
    text_ix = -1

    stack = Stack()
    stack.push(sexpr, 0, id)

    while not stack.empty():
        sexpr, level, parent_id = stack.pop()
        tag, data, is_leaf = tokenize(sexpr)
        id += 1
        if not is_leaf:
            for sexpr in reversed(data):  # reversing here gives desired order
                stack.push(sexpr, level + 1, id)
        else:
            text_ix += 1
        nodes.append(Node(
            tag=tag,
            pos=None,  # don't have it in these sexpr Strings
            token=data[0] if is_leaf else None,
            id=id,
            parent_id=parent_id,
            relationship=None,  # don't have it
            text_ix=text_ix if is_leaf else None,
            level=level,
            is_leaf=is_leaf))

    return Tree(nodes)


# Combining text into internal nodes


def combine_text_at_nodes(tree):
    for l in reversed(range(tree.max_level + 1)):
        nodes = tree.nodes[l]
        for ix in range(len(nodes)):
            node = nodes[ix]
            # for lowest level, set text_at_node to the token
            if l == tree.max_level:
                node.text_at_node = node.token
            # for higher levels, compose these strings
            if l < tree.max_level:
                node.text_at_node = node.token if node.token else ''
                children = [tree.nodes[l+1][cix]
                            for cix in tree.child_ixs[l][ix]]
                sorted_nodes = sorted([node] + [c for c in children],
                                      key=lambda x: x.id)
                nodes_text = [n.text_at_node
                              for n in sorted_nodes
                              if n.text_at_node != '']
                node.text_at_node = ' '.join([tok for tok in nodes_text])
