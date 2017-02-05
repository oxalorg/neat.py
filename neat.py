import math
from collections import defaultdict
import random
import logging
import pprint
print = pprint.PrettyPrinter(width=120).pprint
from enum import Enum

INPUTS = 3
OUTPUTS = 2
innov_no = INPUTS * OUTPUTS - 1

class Layer(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2

def act_fn(z):
    """Sigmoidal activation function"""
    z = min(40, max(-40, z))
    return (2.0 / (1.0 + math.exp(-4.5 * z))) - 1

def create_gene(ip=None, op=None, wt=0.0, enabled=True, innov_no=0):
    gene = {}
    gene['ip'] = ip
    gene['op'] = op
    gene['wt'] = wt
    gene['enabled'] = enabled
    gene['innov_no'] = innov_no
    return gene

def create_neuron(layer=None):
    neuron = {}
    neuron['id'] = 0
    neuron['in_links'] = {}
    neuron['value'] = 0.0
    neuron['type'] = layer
    return neuron

def create_genome():
    genome = {}
    genome['genes'] = {}
    genome['neurons'] = {}
    genome['ip_neurons'] = []
    genome['op_neurons'] = []
    genome['last_neuron'] = 0
    genome['fitness'] = 0.0
    return genome

def init_individual():
    genome = create_genome()

    # Create i/p and o/p neurons
    nid = 0
    for i in range(INPUTS):
        neuron = create_neuron(layer=Layer.INPUT)
        neuron['id'] = nid
        genome['neurons'][nid] = neuron
        genome['ip_neurons'].append(nid)
        nid += 1

    for i in range(OUTPUTS):
        neuron = create_neuron(layer=Layer.OUTPUT)
        neuron['id'] = nid
        genome['neurons'][nid] = neuron
        genome['op_neurons'].append(nid)
        nid += 1

    genome['last_neuron'] = nid - 1
    # Create a gene for every ip, op pair
    innov_no = 0
    for i in range(INPUTS):
        for j in range(OUTPUTS):
            gene = create_gene(innov_no=innov_no)
            gene['ip'] = genome['ip_neurons'][i]
            gene['op'] = genome['op_neurons'][j]
            gene['wt'] = random.random()
            genome['genes'][innov_no] = gene
            #genome['genes'][(gene['ip'], gene['op'])] = gene
            innov_no += 1
    return genome

def create_population(size):
    pop = []
    for i in range(size):
        genome_i = init_individual()
        pop.append(genome_i)
    return pop

def next_innov_no():
    global innov_no
    innov_no += 1
    #generation['innov_no'] += 1
    return innov_no

def next_nid(genome):
    nid = genome['last_neuron'] + 1
    genome['last_neuron'] = nid
    return nid

def mutate(genome):
    NODE_MUTATE_PROB = 0.05
    CONN_MUTATE_PROB = 0.05

    if random.random() < NODE_MUTATE_PROB:
        mutate_add_node(genome)
    if random.random() < CONN_MUTATE_PROB:
        mutate_add_conn(genome)

def mutate_add_conn(g):
    # Select any 2 neurons
    # If they are not connected, connect them
    # Make sure that the the op neuron is not
    # from an input layer
    n1 = random.choice([x for x in g['neurons'].values() if x['type'] != Layer.OUTPUT])
    n2 = random.choice([x for x in g['neurons'].values() if x['type'] != Layer.INPUT])

    nid1 = n1['id']
    nid2 = n2['id']
    if nid1 == nid2:
        return
    # check if a cyclic link exists
    if detect_cycle(g, nid1, nid2):
        logging.debug("Found a cycle")
        return
    if set([(nid1, nid2), (nid2, nid1)]) & set([(x['ip'], x['op']) for x in g['genes'].values()]):
        return

    innov_no = next_innov_no()
    gene = create_gene(ip=nid1, op=nid2, wt=1.0, innov_no=innov_no)
    g['genes'][innov_no] = gene
    logging.debug("mutation: added a conn")

def detect_cycle(g, ip, op):
    if ip == op:
        return

    incoming = defaultdict(list)
    for gene in g['genes'].values():
        incoming[gene['op']].append(gene['ip'])

    unexplored = set([ip])
    explored = set()
    while unexplored:
        node = unexplored.pop()
        explored.add(node)
        for n in incoming[node]:
            if n not in explored:
                unexplored.add(n)
        if op in explored:
            return True
    return False

def mutate_add_node(g):
    # Select any gene
    # Split it into two connections
    gene = random.choice(list(g['genes'].values()))
    if not gene['enabled']:
        logging.debug('Gene was not enabled. Returning.')
        return
    gene['enabled'] = False

    ip, op, wt = gene['ip'], gene['op'], gene['wt']
    neuron = create_neuron(layer=Layer.HIDDEN)
    nid = next_nid(g)
    if ip == op or ip == nid or nid == op:
        logging.error("KILL ME PLS")
        logging.error("{} {} {}".format(ip, nid, op))
        return
    neuron['id'] = nid
    g['neurons'][nid] = neuron

    innov_no1 = next_innov_no()
    innov_no2 = next_innov_no()
    gene1 = create_gene(ip=ip, op=nid, wt=1.0, innov_no=innov_no1)
    gene2 = create_gene(ip=nid, op=op, wt=wt, innov_no=innov_no2)
    g['genes'][innov_no1] = gene1
    g['genes'][innov_no2] = gene2
    logging.debug("mutation: added a node")

def crossover(mom, dad):
    if mom['fitness'] < dad['fitness']:
        dad, mom = mom, dad

    child = create_genome()
    child['neurons'].update(mom['neurons'])
    child['neurons'].update(dad['neurons'])

    # checks with innovation numbers (which are keys of the gene dict)
    for gene in mom['genes']:
        if gene in dad['genes'] and random.random() < 0.5:
            # matching gene
            child['genes'][gene] = dad['genes'][gene].copy()
        else:
            child['genes'][gene] = mom['genes'][gene].copy()

    for gene in dad['genes']:
        if gene not in mom['genes']:
            child['genes'][gene] = dad['genes'][gene].copy()

    last_neuron = max([x['id'] for x in child['neurons'].values()])
    child['last_neuron'] = last_neuron
    return child

def create_layers(g):
    nodep = {x for x in g['ip_neurons']}
    layers = [nodep.copy()]
    remaining = {x for x in g['neurons'].keys()} - nodep
    incoming = defaultdict(list)
    for gene in g['genes'].values():
        incoming[gene['op']].append(gene['ip'])
    while True:
        L = set()
        for node in remaining:
            if set(incoming[node]) <= nodep:
                nodep.add(node)
                L.add(node)

        if not L:
            logging.error("Circular dependency exists")
            print(remaining)
            break

        layers.append(L)
        for node in L:
            remaining.remove(node)

        if not remaining:
            break

    return layers

def generate_network(g):
    layers = create_layers(g)
    def activate(inputs):
        # set the values for the inputs
        values = {x: 0.0 for x in g['neurons'].keys()}
        for i, ip_n in enumerate(g['ip_neurons']):
            #g['neurons'][ip_n]['value'] = inputs[i]
            values[ip_n] = inputs[i]

        incoming = defaultdict(list)
        wt = {}
        for gene in g['genes'].values():
            incoming[gene['op']].append(gene['ip'])
            wt[(gene['ip'], gene['op'])] = gene['wt']

        for layer in layers[1:]:
            for node in layer:
                total = 0
                for ip in incoming[node]:
                    total += wt[(ip, node)] * values[ip]
                total = act_fn(total)
                values[node] = total
                print(values)

        for op_n in g['op_neurons']:
            print(op_n)
            print(values[op_n])

    return activate

def main():
    pop_size = 20
    pop = create_population(pop_size)
    for _ in range(4000):
        mutate(pop[0])
    print(pop[0])
    nw = generate_network(pop[0])
    nw([-0.55, 0.4, -0.05])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
