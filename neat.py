import math
import time
import argparse
import copy
import sys
from collections import defaultdict
import random
import logging
import pprint
pprint = pprint.PrettyPrinter(width=120).pprint
from enum import Enum
logging.basicConfig(level=logging.ERROR)

# Number of i/p, o/p and bias units.
INPUTS = 2
OUTPUTS = 1
BIAS = 1
# Global innovation number, all individuals populations
# are assumed to have same historical origins
# hence their innovation numbers are same for the intially
# fully connected network starting from 0 upto ``innov_no``
innov_no = (INPUTS + BIAS) * OUTPUTS - 1

class Layer(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    BIAS = 3

def act_fn(z):
    """
    Sigmoidal activation function
    """
    z = min(40, max(-40, z))
    return 1.0 / (1.0 + math.exp(-4.9 * z))

def create_gene(ip=None, op=None, wt=0.0, enabled=True, innov_no=0):
    """
    Create a simple base gene i.e. a connection/synapse
    """
    gene = {}
    gene['ip'] = ip
    gene['op'] = op
    gene['wt'] = wt
    gene['enabled'] = enabled
    gene['innov_no'] = innov_no
    return gene

def create_neuron(layer=None):
    """
    Create a simple base neuron i.e. a node
    """
    neuron = {}
    neuron['id'] = 0
    #neuron['in_links'] = {}
    #neuron['value'] = 0.0
    neuron['type'] = layer
    return neuron

def create_genome():
    """
    Create a simple base genome i.e. a genotype
    """
    genome = {}
    genome['genes'] = {}
    genome['neurons'] = {}
    genome['ip_neurons'] = []
    genome['op_neurons'] = []
    genome['bias_neurons'] = []
    genome['last_neuron'] = 0
    genome['fitness'] = 0.0
    return genome

def copy_genome(genome):
    """
    Fast copy a genome
    """
    clone = {}
    #clone['genes'] = genome['genes'].copy()
    #clone['neurons'] = genome['neurons'].copy()
    clone['genes'] = copy.deepcopy(genome['genes'])
    clone['neurons'] = copy.deepcopy(genome['neurons'])
    clone['ip_neurons'] = genome['ip_neurons']
    clone['op_neurons'] = genome['op_neurons']
    clone['bias_neurons'] = genome['bias_neurons']
    clone['last_neuron'] = genome['last_neuron']
    clone['fitness'] = genome['fitness']
    return clone

def init_individual():
    """
    Creates an individual with all I/P, O/P fully
    connected, and BIAS connected to all O/Ps

    returns a genome
    """
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
    """
    Creates a set of fully connected individuals
    """
    pop = []
    for i in range(size):
        genome_i = init_individual()
        pop.append(genome_i)
    return pop

def next_innov_no():
    """
    Tracker for global innovations among genes
    """
    global innov_no
    innov_no += 1
    #generation['innov_no'] += 1
    return innov_no

def next_nid(genome):
    """
    Tracker for next neuron id in the given genome
    """
    nid = genome['last_neuron'] + 1
    genome['last_neuron'] = nid
    return nid

def mutate(genome):
    """
    Given a genome, mutates it in-place
    """
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
    """
    Mates 2 individuals and returns an offspring
    """
    # make sure the fitter parent is mom
    if mom['fitness'] < dad['fitness']:
        dad, mom = mom, dad

    # create a new child and copy over all the 
    # information except for genes
    # from mom (the fitter parent)
    child = copy_genome(mom)
    child['genes'] = {}

    # Copy genes from both parents to the child
    # We use historical markings i.e. the innovation 
    # numbers (which are keys of the genes dict)
    for gene in mom['genes']:
        if gene in dad['genes'] and random.random() < 0.5:
            # matching gene is copied from either parents with a probability
            child['genes'][gene] = dad['genes'][gene].copy()
        else:
            # disjoint gene, copy from the fitter parent
            child['genes'][gene] = mom['genes'][gene].copy()

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


def adjusted_pop_size(species, pop_size):
    """
    Finds the adjusted, normalized population size
    for the next generation of the given species
    """
    size = {}
    sp_fitness = {}
    avg_wt = 0
    #total_avg

    #for sp in species.values():
        

    for i, sp in species.items():
        sp_fitness[i] = sum([x['fitness'] for x in sp['members']])
        sp_fitness[i] /= len(sp['members'])
        avg_wt += sp_fitness[i]

    avg_wt /= len(species)

    for i, sp in species.items():
        sp_size = len(sp['members'])
        sp_size = sp_fitness[i] / avg_wt * pop_size
        if sp_fitness[i] > avg_wt:
            sp_size *= 1.08
        else:
            sp_size *= 0.93
        size[i] = sp_size
        #print("Curr size: ", len(sp['members']))
        #print("Expe size: ", size[i])

    # normalize
    total_size = sum(size.values())
    for i in size:
        size[i] *= pop_size / total_size
        size[i] = int(size[i])

    return size


def remove_stagnant(species):
    """
    If the average adjusted fitness of a species
    has not changed since the past 15 generations
    remove it from the species pool
    """
    stagnant = []
    for i, sp in species.items():
        sp_fitness = sum([x['fitness'] for x in sp['members']])
        if sp_fitness > sp['prev_fitness']:
            sp['stag_count'] = 0
        else:
            sp['stag_count'] += 1

        if sp['stag_count'] >= 15:
            stagnant.append(i)

        sp['prev_fitness'] = sp_fitness

    logging.info("Stagnant species this generations: {}".format(stagnant))
    for i in stagnant:
        del species[i]


def fitness_sharing(species):
    """
    Performs explicit fitness sharing in-place
    """
    for i, sp in species.items():
        members = sp['members']
        n = len(members)
        for g in members:
            g['adj_fitness'] = g['fitness']/n


def reproduce(species, pop_size):
    """
    Given a list of individuals, perform mating
    and mutation to return individuals for newer generation
    """
    new_pop = []
    adj_ftn = {}
    avg_adj_ftn = 0

    remove_stagnant(species)
    fitness_sharing(species)
    new_pop_size = adjusted_pop_size(species, pop_size)

    for i, sp in species.items():
        size = new_pop_size[i]

        if size == 0:
            continue

        # remove 25% most unfit members
        mem_size = len(sp['members'])
        members = sorted(sp['members'], key=lambda x: x['fitness'])[int(mem_size)//4:]

        if len(members) > 0:
            # If the species has atleast 1 individuals
            # copy the fittest individual as it is
            new_pop.append(members[-1])
            size -= 1

        norm_size = int(size)
        norm_60 = int(norm_size * 0.60)
        norm_25 = int(norm_size * 0.25)
        # account for loss due to rounding
        norm_15 = norm_size - norm_60 - norm_25


        # 15% of the new population is obtained from mutations
        # of the best 10% of the species
        best_10 = members[-int(mem_size * 0.1):]
        for i in range(norm_15):
            child = copy_genome(random.choice(best_10))
            #pprint(child)
            mutate(child)
            new_pop.append(child)

        # 60% of the new population is from same species mating
        for i in range(norm_size):
            dad = random.choice(members)
            mom = random.choice(members)
            child = crossover(dad, mom)
            mutate(child)
            new_pop.append(child)

        # The remaining 25% are pure mutations
        for i in range(norm_25):
            child = copy_genome(random.choice(members))
            mutate(child)
            new_pop.append(child)

    return new_pop


def update_reps(species):
    for i, sp in species.items():
        sp['rep'] = sp['members'][0].copy()


def calc_DEW(g1, g2):
    gene1_set = {x for x in g1['genes']}
    gene2_set = {x for x in g2['genes']}
    excess_marker = max(gene1_set)

    complete = gene1_set | gene2_set
    matching = gene1_set & gene2_set
    avg_wt = 0
    for gene in matching:
        avg_wt += abs(g1['genes'][gene]['wt'] - g2['genes'][gene]['wt'])
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


def empty_species(species):
    for sp in species.values():
        sp['members'] = []


def speciate(new_pop, species):
    delta_th = 3.0
    for g in new_pop:
        for i, sp in species.items():
            rep = sp['rep']
            if delta_fn(g, rep) < delta_th:
                sp['members'].append(g)
                break
        else:
            next_sp_id = max(species.keys(), key=int)
            species[next_sp_id+1] = { 'members': [g],
                                        'rep': g,
                                        'stag_count': 0,
                                        'prev_fitness': sys.float_info.min }

    # kill empty species and convert to a list
    empty = []
    for i, sp in species.items():
        if not len(sp['members']):
            empty.append(i)

    logging.info("Empty species this generation: {}".format(empty))
    for e in empty: del species[e]


def print_fittest(species, verbose=False, compact=False, file=sys.stdout):
    """
    prints information regard the fittest individual in all species
    to a file or stdout

    returns the fittest genome
    """
    fittest = []
    for sp in species.values():
        members = sp['members']
        fittest.append(max([x for x in members], key=lambda x: x['fitness']))
    fit = max([x for x in fittest], key= lambda x: x['fitness'])

    if verbose:
        print("Fitness: {:.03f}, Genes: {}, Neurons: {}".format(fit['fitness'], len(fit['genes']), len(fit['neurons'])))
        print("Species len: {}".format(len(species)))
    print("Fitness: {:.03f}, Genes: {}, Neurons: {}".format(fit['fitness'], len(fit['genes']), len(fit['neurons'])), file=file)
    print("Species len: {}".format(len(species)), file=file)

#    if not compact:
#        nw = generate_network(fit)
#        for xi, xo in zip(xor_inputs, xor_outputs):
#            output = nw(xi)
#            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output), file=file)
    return fit


def main(fitness, gen_size=100, pop_size=150, verbose=False, fitness_thresh=None):
    pop = create_population(pop_size)
    fitness(pop)
    yield pop
    species = { 0: { 'members': pop,
                      'rep': pop[0],
                      'stag_count': 0,
                      'prev_fitness': sys.float_info.min } }

    slen = []

    fp = open('log.txt', 'w')
    for gen in range(gen_size):
        sys.stdout.write("Generation {}\r".format(gen))

        # primary loop for generations
        # 1. get the representatives of the species for
        # 2. reproduce the species to get back a new population
        # 3. update the fitness value of the population
        # 4. empty the current list of species members
        # 5. speciate the population into their own species
        update_reps(species)
        t1 = time.time()
        pop = reproduce(species, pop_size)
        t2 = time.time()
        logging.warning("Reproduction took: {:0.02f} seconds".format(t2 - t1))
        yield pop
        fitness(pop)
        t3 = time.time()
        logging.warning("Fitness took: {:0.02f} seconds".format(t3 - t2))
        empty_species(species)
        speciate(pop, species)
        t4 = time.time()
        logging.warning("Speciation took: {:0.02f} seconds".format(t4 - t3))

        slen.append(len(species))
        fittest = print_fittest(species, verbose=verbose, file=fp)
        if fitness_thresh and abs(fittest['fitness']) > fitness_thresh:
            print("Fitness threshold reached")
            break

    fp.close()
    fit = print_fittest(species)
    print("+===========+FITTEST SURVIOR+=============+")
    pprint(fit)
    yield fit
    #setlen = set(slen)
    #pprint([(i, slen.count(i)) for i in setlen])


def cli():
    parser = argparse.ArgumentParser(
            description='neat.py - My personal implementation of NEAT',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g',
            '--generations',
            dest='generations',
            type=int,
            default=100,
            help='Number of generations')
    parser.add_argument('-p',
            '--population',
            dest='population',
            type=int,
            default=150,
            help='Population size')
    parser.add_argument('-v',
            '--verbose',
            dest='verbose',
            action='store_true',
            help='Prints some information to stdout')
    args = parser.parse_args()
    main(args.generations, args.population, args.verbose)


if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        sys.exit()

