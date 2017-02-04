import math
import random
import logging

class Genome:
    def __init__(self):
        self.gid = 0
        self.cid = 0
        self.conn = set()
        self.nodes = set()
        self.links = set()
        self.inputs = []
        self.outputs = []

    def get_gid(self):
        self.gid += 1
        return self.gid

    def get_cid(self):
        self.cid += 1
        return self.cid


class Node:
    def __init__(self, genome):
        self.id = genome.get_gid()
        self.in_links = set()
        self.out_links = set()
        genome.nodes.add(self)

    def __repr__(self):
        return "{}".format(self.id)


class Connection:
    def __init__(self, in_node, out_node, wt, genome, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.wt = wt
        self.enabled = enabled
        self.id = genome.get_cid()
        genome.conn.add(self)
        genome.links.add((in_node, out_node))
        in_node.out_links.add(self)
        out_node.in_links.add(self)

    def __repr__(self):
        return "{}: ({} -- {} --> {}) [{}]".format(self.id, self.in_node, self.wt, self.out_node, self.enabled)


def cyclic_move(genome, i, o):
    """
    Checks if the new link (i --> o) creates a cycle in the
    given genome.
    """
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in genome.links:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def mutate(genome):
    """
    Mutates the given genome inplace. Returns nothing.
    """
    STRUCT_MUTATE_RATE = 0.05
    CONN_MUTATE_RATE = 0.02

    if random.random() < STRUCT_MUTATE_RATE:
            logging.info("Mutation: adding a node")
            # add node - an existing connection is split and the new node placed there
            # old connection is disabled
            # 2 new connections are added to the genome
            # new connection out of <--new node gets weight of old connection
            # new connection leading into -->new node gets weight 1.0
            # This method minimizes initial effect of mutation
            node = Node(genome)
            old_conn = random.choice(tuple(genome.conn))
            n1 = old_conn.in_node
            n2 = old_conn.out_node
            old_conn.enabled = False
            new_conn1 = Connection(node, n2, old_conn.wt, genome)
            new_conn2 = Connection(n1, node, 1.0, genome)

    if random.random() < STRUCT_MUTATE_RATE:
            logging.info("Mutation: adding a connection")
            # add connection
            # new connection gene with random wt is added,
            # connecting 2 previously unconnected nodes
            illegal_nodes = set(genome.inputs)
            for selected_node in set(random.sample(genome.nodes, k=len(genome.nodes))) - set(genome.outputs):
                for node in genome.nodes - set([selected_node]) - illegal_nodes:
                    out_connected = set(x.out_node for x in node.out_links)
                    for n in genome.nodes - out_connected - illegal_nodes:
                        if not cyclic_move(genome, node, n):
                            new_conn = Connection(node, n, random.random(), genome)
                            return
            else:
                logging.info("Mutation: could not find any legal connections to add")

    for conn in genome.conn:
        if random.random() < CONN_MUTATE_RATE:
            conn.wt = random.random()

def act_fn(z):
    """Sigmoidal activation function"""
    z = max(-60.0, min(60.0, 5.0 * z))
    return (2.0 / (1.0 + math.exp(-z))) - 1


def gen_network(genome):
    """
    Given a genome, creates a feed forward nerual network.
    Returns an ``activate`` function which returns the desired
    output on the given inputs.
    """
    layers = create_layers(genome)

    def activate(ip):
        net = {v: ip[i] for i, v in enumerate(genome.inputs)}

        # Don't include input layer for iteration
        # since we already have the required values for i/p layer
        for layer in layers[1:]:
            for node in layer:
                # find all nodes on the other side of in_links
                # if the connection is enabled, add the net output
                # of the other node multiplied by connection wt
                net_node = []
                for n, w, e in set((x.in_node, x.wt, x.enabled) for x in node.in_links):
                    if e:
                        net_node.append(w * net[n])
                net[node] = act_fn(sum(net_node))

        op = [net.get(out, 0) for out in genome.outputs]
        return op

    return activate


def create_layers(genome):
    """
    Creates a list of layers for the given genome.
    each layer is a set of nodes.
    """
    # Starting from first layer of inputs, it successively
    # finds all those nodes which have their inlinks contained
    # in the (i-1)th layer.
    visited = set(genome.inputs)
    layers = [genome.inputs]
    while True:
        next_layer = set()
        for node in genome.nodes - visited:
            if set(x.in_node for x in node.in_links) <= visited:
                next_layer.add(node)

        if not next_layer:
            break

        visited |= next_layer
        layers.append(next_layer)

    return layers


def test():
    g = Genome()

    n1 = Node(g)
    n2 = Node(g)
    n3 = Node(g)
    n4 = Node(g)
    n5 = Node(g)
    n6 = Node(g)

    g.inputs = [n1, n2, n3]
    g.outputs = [n5]

    c1 = Connection(n1, n5, 0.8, g)
    c2 = Connection(n2, n4, 0.5, g)
    c3 = Connection(n3, n4, 0.1, g)
    c4 = Connection(n3, n5, 0.0, g)
    c5 = Connection(n4, n5, 1.0, g)
    c6 = Connection(n5, n6, 1.0, g)

    g.nodes = g.nodes | set([n1, n2, n3, n4, n5, n6])

    # for x in g.conn:
    #     print(x)

    # layers = create_layers(g, inputs, outputs)
    nw = gen_network(g)
    print(nw([0.5,0.1,0.2]))
    print(nw([-0.1,-0.2,0.1]))
    for x in g.conn:
        print(x)
    mutate(g)
    print('---')
    for x in g.conn:
        print(x)


def main():
    pop_size = 10
    input_size = 2
    output_size = 1
    genomes = []
    for _ in range(pop_size):
        g = Genome()
        genomes.append(g)
        for i in range(input_size):
            g.inputs.append(Node(g))
        for i in range(output_size):
            g.outputs.append(Node(g))

        for i in g.inputs:
            for j in g.outputs:
                Connection(i, j, random.random(), g)

        nw = gen_network(g)
        print(nw([0.5,-0.9]))
        print(nw([-0.1,0.1]))
        for _ in range(50):
            mutate(g)
        nw = gen_network(g)
        print(nw([0.5,-0.9]))
        print(nw([-0.1,0.1]))
        print('---')



if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
    #test()
