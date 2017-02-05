import math
import random
import logging

class Genome:
    innov_no = 0
    innov_tracker = {'node': {}, 'conn': {}}
    def __init__(self):
        self.gid = 0
        self.cid = 0
        self.conn = set()
        self.nodes = set()
        self.links = set()
        self.threads = dict()
        self.inputs = []
        self.outputs = []
        self.fitness = 0

    def get_gid(self):
        self.gid += 1
        return self.gid

    def get_cid(self):
        self.cid += 1
        return self.cid

    @classmethod
    def get_innov_no(self):
        Genome.innov_no += 1
        return Genome.innov_no

class Node:
    def __init__(self, genome, id=None):
        if id:
            self.id = id
        else:
            self.id = genome.get_gid()
        self.in_links = set()
        self.out_links = set()
        genome.nodes.add(self)

    def __repr__(self):
        return "{}".format(self.id)


class Connection:
    def __init__(self, in_node, out_node, wt, genome, innov_no=None, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.wt = wt
        self.enabled = enabled
        #self.id = genome.get_cid()
        if not innov_no:
            self.innov_no = Genome.get_innov_no()
        else:
            self.innov_no = innov_no
        genome.conn.add(self)
        genome.links.add((in_node, out_node))
        genome.threads[(in_node.id, out_node.id)] = self
        in_node.out_links.add(self)
        out_node.in_links.add(self)

    def __repr__(self):
        return "{}: ({} --> {}) [{}]".format(self.innov_no, self.in_node, self.out_node, self.enabled)


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


def crossover(p1, p2):
    """
    Performs crossover on 2 parents p1, p2.
    Returns their offspring.
    """
    if p1.fitness < p2.fitness:
        p1, p2 = p2, p1

    offspring = Genome()
    matching1 = set()
    matching2 = set()

    for _ in p1.inputs:
        offspring.inputs.append(Node(offspring))
    for _ in p1.outputs:
        offspring.outputs.append(Node(offspring))

    for _ in p1.nodes | p2.nodes - set(p1.inputs) - set(p1.outputs):
        Node(offspring)

    for conn1 in p1.conn:
        for conn2 in p2.conn:
            if conn1.innov_no == conn2.innov_no:
                # matching genes
                matching1.add(conn1)
                matching2.add(conn2)
                Connection(conn1.in_node, conn1.out_node, conn1.wt, offspring, conn1.innov_no, enabled=conn1.enabled)

    for conn in (p1.conn - matching1) | (p2.conn - matching2):
        # non matching genes
        Connection(conn.in_node, conn.out_node, conn.wt, offspring, conn.innov_no, enabled=conn.enabled)

    return offspring


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
        old_conn = random.choice(tuple(genome.conn))
        innov_no1 = None
        innov_no2 = None
        id = None
        n1 = old_conn.in_node
        n2 = old_conn.out_node
        marker = (n1.id, n2.id)
        if marker in Genome.innov_tracker['node']:
            history = Genome.innov_tracker['node'][marker]
            id = history[0]
            innov_no1 = history[1]
            innov_no2 = history[2]
        node = Node(genome, id)
        old_conn.enabled = False
        new_conn1 = Connection(node, n2, old_conn.wt, genome, innov_no1)
        new_conn2 = Connection(n1, node, 1.0, genome, innov_no2)
        if not id:
            Genome.innov_tracker['node'][marker] = (node.id, new_conn1.innov_no, new_conn2.innov_no)

    if random.random() < STRUCT_MUTATE_RATE - 1:
        logging.info("Mutation: adding a connection")
        # add connection
        # new connection gene with random wt is added,
        # connecting 2 previously unconnected nodes
        illegal_nodes = set(genome.inputs)
        for selected_node in set(random.sample(genome.nodes, k=len(genome.nodes))) - set(genome.outputs):
            for node in genome.nodes - set([selected_node]) - illegal_nodes:
                out_connected = set(x.out_node for x in node.out_links)
                for n in genome.nodes - out_connected - illegal_nodes:
                    if not cyclic_move(genome, node, n) and not (node, n) in genome.threads and not (n, node) in genome.threads:
                        innov_no = None
                        marker = (node.id, n.id)
                        if marker in Genome.innov_tracker['conn']:
                            history = Genome.innov_tracker['conn'][marker]
                            innov_no = history
                        new_conn = Connection(node, n, random.random(), genome, innov_no)
                        if not innov_no:
                            Genome.innov_tracker['conn'][marker] = new_conn.innov_no
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
    p = Genome()

    n1 = Node(g)
    n2 = Node(g)
    n3 = Node(g)

    p1 = Node(p)
    p2 = Node(p)
    p3 = Node(p)

    g.inputs = [n1, n2]
    g.outputs = [n3]
    p.inputs = [n1, n2]
    p.outputs = [n3]

    c1 = Connection(n1, n3, 0.8, g)
    c2 = Connection(n2, n3, 0.5, g)
    c1 = Connection(p1, p3, 0.8, p)
    c2 = Connection(p2, p3, 0.5, p)

    g.nodes = g.nodes | set([n1, n2, n3])
    p.nodes = p.nodes | set([p1, p2, p3])

    mutate(g)


def main():
    pop_size = 6
    input_size = 2
    output_size = 1
    genomes = []
    Genome.innov_no += input_size + output_size
    for _ in range(pop_size):
        innov_no = 1
        g = Genome()
        genomes.append(g)
        for i in range(input_size):
            g.inputs.append(Node(g))
        for i in range(output_size):
            g.outputs.append(Node(g))

        for i in g.inputs:
            for j in g.outputs:
                Connection(i, j, random.random(), g, innov_no)
                innov_no += 1

        for _ in range(30):
            mutate(g)


    print('Conn 1')
    print(genomes[0].conn)
    print(genomes[0].threads)
    for x in genomes[0].conn:
        print(x)
    print('Conn 2')
    for x in genomes[1].conn:
        print(x)
    print('~')
    offspring = crossover(genomes[0], genomes[1])
    for x in offspring.conn:
        print(x)
    print('~')


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
    #test()
