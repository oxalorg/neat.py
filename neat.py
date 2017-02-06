import math
import sys
from collections import defaultdict
import random
import logging
import pprint
pprint = pprint.PrettyPrinter(width=120).pprint
from enum import Enum

INPUTS = 2
OUTPUTS = 1
BIAS = 1
innov_no = INPUTS * OUTPUTS - 1

class Layer(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    BIAS = 3

def act_fn(z):
    """Sigmoidal activation function"""
    z = min(40, max(-40, z))
    return 1.0 / (1.0 + math.exp(-4.9 * z))
    #return (2.0 / (1.0 + math.exp(-4.5 * z))) - 1

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
    #neuron['in_links'] = {}
    #neuron['value'] = 0.0
    neuron['type'] = layer
    return neuron

def create_genome():
    genome = {}
    genome['genes'] = {}
    genome['neurons'] = {}
    genome['ip_neurons'] = []
    genome['op_neurons'] = []
    genome['bias_neurons'] = []
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

    for i in range(BIAS):
        neuron = create_neuron(layer=Layer.BIAS)
        neuron['id'] = nid
        genome['neurons'][nid] = neuron
        genome['bias_neurons'].append(nid)
        nid += 1

    genome['last_neuron'] = nid - 1
    # Create a gene for every ip, op pair
    innov_no = 0
    for i in range(INPUTS):
        for j in range(OUTPUTS):
            gene = create_gene(innov_no=innov_no)
            gene['ip'] = genome['ip_neurons'][i]
            gene['op'] = genome['op_neurons'][j]
            gene['wt'] = random.random() * 2 - 1
            genome['genes'][innov_no] = gene
            #genome['genes'][(gene['ip'], gene['op'])] = gene
            innov_no += 1

    for i in range(BIAS):
        for j in range(OUTPUTS):
            gene = create_gene(innov_no=innov_no)
            gene['ip'] = genome['bias_neurons'][i]
            gene['op'] = genome['op_neurons'][j]
            gene['wt'] = random.random() * 2 - 1
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
    NODE_MUTATE_PROB = 0.03
    CONN_MUTATE_PROB = 0.05
    WT_MUTATE_PROB = 0.8
    WT_PERTURBED_PROB = 0.9

    if random.random() < NODE_MUTATE_PROB:
        mutate_add_node(genome)
    if random.random() < CONN_MUTATE_PROB:
        mutate_add_conn(genome)
    if random.random() < WT_MUTATE_PROB:
        for gene in genome['genes'].values():
            if random.random() < WT_PERTURBED_PROB:
                gene['wt'] = gene['wt'] * (1 + (random.random() * 2 - 1)/10)
            else:
                gene['wt'] = random.random() * 2 - 1


def mutate_add_conn(g):
    # Select any 2 neurons
    # If they are not connected, connect them
    # Make sure that the the op neuron is not
    # from an input layer
    n1 = random.choice([x for x in g['neurons'].values() if x['type'] != Layer.OUTPUT])
    n2 = random.choice([x for x in g['neurons'].values() if x['type'] != Layer.INPUT and x['type'] != Layer.BIAS])

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
        return False

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
    child['ip_neurons'] = mom['ip_neurons']
    child['op_neurons'] = mom['op_neurons']
    child['bias_neurons'] = mom['bias_neurons']
    child['neurons'].update(mom['neurons'])
    child['neurons'].update(dad['neurons'])

    # checks with innovation numbers (which are keys of the gene dict)
    for gene in mom['genes']:
        if gene in dad['genes'] and random.random() < 0.5:
            # matching gene
            child['genes'][gene] = dad['genes'][gene].copy()
        else:
            # disjoint gene, copy from fitter parent
            child['genes'][gene] = mom['genes'][gene].copy()

    #for gene in dad['genes']:
    #    if gene not in mom['genes']:
    #        child['genes'][gene] = dad['genes'][gene].copy()

    last_neuron = max([x['id'] for x in child['neurons'].values()])
    child['last_neuron'] = last_neuron + 1
    return child

def create_layers(g):
    nodep = {x for x in g['ip_neurons']}
    layers = [nodep.copy()]
    remaining = {x for x in g['neurons'].keys()} - nodep
    incoming = defaultdict(list)
    wt = {}
    for gene in g['genes'].values():
        incoming[gene['op']].append(gene['ip'])
        wt[(gene['ip'], gene['op'])] = gene['wt']
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

    return layers, incoming, wt

def generate_network(g):
    layers, incoming, wt = create_layers(g)

    def activate(inputs):
        # set the values for the inputs
        values = {x: 0.0 for x in g['neurons'].keys()}
        for i, ip_n in enumerate(g['ip_neurons']):
            #g['neurons'][ip_n]['value'] = inputs[i]
            values[ip_n] = inputs[i]

        values[g['bias_neurons'][0]] = 1.0

        for layer in layers[1:]:
            for node in layer:
                total = 0
                if not incoming[node]:
                    # if no incoming node, don't apply actv function
                    continue
                for ip in incoming[node]:
                    total += wt[(ip, node)] * values[ip]
                total = act_fn(total)
                values[node] = total

        outputs = [values[op] for op in g['op_neurons']]
        return outputs

    return activate

def reproduce(species):
    """
    Given a list of individuals, perform mating
    and mutation to return individuals for newer generation
    """
    new_pop = []
    adj_ftn_sum = 0
    for sp in species:
        adj_ftn_sum += sum([x['fitness'] for x in sp])
    for sp in species:
        sp_size = len(sp)
        # remove 25% most unfit members
        sp = sorted(sp, key=lambda x: x['fitness'])[sp_size//4:]
        norm_sp_size = sum([x['fitness'] for x in sp])//adj_ftn_sum
        new_pop.append(sp.pop())
        norm_sp_size = sp_size - 1
        norm_25 = norm_sp_size // 4
        while norm_sp_size > 0 and norm_sp_size > norm_25:
            dad = random.choice(sp)
            mom = random.choice(sp)
            child = crossover(dad, mom)
            mutate(child)
            new_pop.append(child)
            norm_sp_size -= 1
        while norm_sp_size > 0:
            #print("W0000000000000000000T1231231000000000000000000000000000000000000")
            child = random.choice(sp)
            mutate(child)
            new_pop.append(child)
            norm_sp_size -= 1

    return new_pop

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def get_reps(species):
    reps = [sp[0] for sp in species]
    return reps

def fitness(pop):
    """
    Recieves a list of pop. Modify ONLY their
    fitness values
    """
    for g in pop:
        g['fitness'] = 4.0
        nw_activate = generate_network(g)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = nw_activate(xi)
            g['fitness'] -= (output[0] - xo[0]) ** 2

def calc_DEW(g1, g2):
    gene1_set = {x for x in g1['genes']}
    gene2_set = {x for x in g2['genes']}
    excess_marker = max(gene1_set)

    complete = gene1_set | gene2_set
    matching = gene1_set & gene2_set
    avg_wt = 0
    for gene in matching:
        avg_wt += g1['genes'][gene]['wt'] - g2['genes'][gene]['wt']
    avg_wt /= len(matching)

    non_matching = complete - matching
    excess = len([x for x in non_matching if x > excess_marker])
    disjoint = len([x for x in non_matching if x <= excess_marker])
    return disjoint, excess, avg_wt

def delta_fn(g1, g2):
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4
    N = max(len(g1['genes']), len(g2['genes']))
    N = 1 if N < 20 else N
    d, e, w = calc_DEW(g1, g2)
    delta = (c2 * d + c1 * e)/N + c3 * w
    return delta

def speciate(pop, reps):
    delta_th = 3.0
    species = [(rep, []) for rep in reps]
    for g in pop:
        for sp in species:
            if delta_fn(g, sp[0]) < delta_th:
                sp[1].append(g)
                break
        else:
            species.append((g, [g]))

    # explicit fitness sharing
    for sp in species:
        n = len(sp[1])
        for g in sp[1]:
            g['fitness'] = g['fitness']/n

    # kill empty species and convert to a list
    sp_list = [x[1] for x in species if x[1]]

    return sp_list


def print_fittest(species, file=sys.stdout):
    fittest = []
    for sp in species:
        fittest.append(max([x for x in sp], key=lambda x: x['fitness']))
    fit = max([x for x in fittest], key= lambda x: x['fitness'])
    print("Fitness: {:.03f}, Genes: {}, Neurons: {}".format(fit['fitness'], len(fit['genes']), len(fit['neurons'])), file=file)
    print("Species len: {}".format(len(species)), file=file)
    nw = generate_network(fit)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = nw(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output), file=file)


def main():
    pop_size = 100
    pop = create_population(pop_size)
    fitness(pop)
    species = [pop]
    reps = [pop[0]]

    slen = []
    fp = open('log.txt', 'w')
    for gen in range(50):
        # primary loop for generations
        # 1. get the representatives of the species for
        #    the current generation
        # 2. reproduce the species to get back a new population
        # 3. update the fitness value of the population
        # 4. speciate the population into their own species
        sys.stdout.write("Generation {}\r".format(gen))
        reps = get_reps(species)
        pop = reproduce(species)
        fitness(pop)
        species = speciate(pop, reps)
        slen.append(len(species))
        print_fittest(species, fp)

    fp.close()
    print_fittest(species)
    #setlen = set(slen)
    #pprint([(i, slen.count(i)) for i in setlen])

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    try:
        for i in range(1):
            main()
    except KeyboardInterrupt:
        sys.exit()

