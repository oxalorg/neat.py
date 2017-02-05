import math
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
    mutate_add_node(genome)
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
    # check if a link already exsits
    if set([(nid1, nid2), (nid2, nid1)]) & set([(x['ip'], x['op']) for x in g['genes'].values()]):
        return

    innov_no = next_innov_no()
    gene = create_gene(ip=nid1, op=nid2, wt=1.0, innov_no=innov_no)
    g['genes'][innov_no] = gene

def mutate_add_node(g):
    # Find a random gene
    # Split it into two connections
    gene = random.choice(list(g['genes'].values()))
    if not gene['enabled']:
        logging.debug('Gene was not enabled. Returning.')
        return
    gene['enabled'] = False

    ip, op, wt = gene['ip'], gene['op'], gene['wt']
    neuron = create_neuron(layer=Layer.HIDDEN)
    nid = next_nid(g)
    neuron['id'] = nid
    g['neurons'][nid] = neuron

    innov_no1 = next_innov_no()
    innov_no2 = next_innov_no()
    gene1 = create_gene(ip=ip, op=nid, wt=1.0, innov_no=innov_no1)
    gene2 = create_gene(ip=nid, op=op, wt=wt, innov_no=innov_no2)
    g['genes'][innov_no1] = gene1
    g['genes'][innov_no2] = gene2

def main():
    ip = 3
    op = 2
    pop_size = 4
    pop = create_population(pop_size)
    print(pop[0])
    mutate(pop[0])
    print("~!@~!@~!@~!@~!@~!@~!@~!@~!@~!@~!@~!@")
    print(pop[0])

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
