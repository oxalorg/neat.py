import random
import os
import neat

def find_score(s):
    for i in range(s):
        if s[i+1] == '-' and s[i] != 'J':
            return i
    return i

def block_ahead(s, i):
    return s[i+1] == '1' and s[i] == '0'


NUM_TESTS = 4
# STR_INPUT = ['000', '001', '010']
STR_INPUT = ['000', '002', '010']
# STR_INPUT = ['___', '__-', '_-_']
STR_LEN = 10

TRAINING_SET = [''.join(random.choice(STR_INPUT) for _ in range(STR_LEN)) for _ in range(NUM_TESTS)]

def eval_fitness(genomes):
    for g in genomes:
        net = neat.generate_network(g)
        err = 0.0
        for t in range(NUM_TESTS):
            seq = TRAINING_SET[t]
            out = ''
            for i in range(len(seq)-1):
                inputs = [int(seq[i]), int(seq[i+1])]
                output = net(inputs)
                #print(output)
                #print(inputs)
                if block_ahead(seq, i):
                    expected = 1
                else:
                    expected = 0
                err += (expected - output[0] ) ** 2
                # print(output)
        # print("ERROR IS : ", err)
        g['fitness'] = NUM_TESTS*STR_LEN - err


def eval_fitness2(genomes):
    for g in genomes:
        net = neat.generate_network(g)
        err = -1
        g.fitness = g.fitness or 0
        for t in range(NUM_TESTS):
            seq = ''.join(random.choice(STR_INPUT) for _ in range(STR_LEN))
            out = ''
            dead = False
            for i in range(len(seq)-1):
                inputs = [int(seq[i]), int(seq[i+1])]
                output = net.serial_activate(inputs)
                #print(output)
                #print(inputs)
                if dead:
                    break
                if block_ahead(seq, i) and output[0] < 0.5:
                    dead = True
                    g.fitness = g.fitness
                else:
                    g.fitness += 1


local_dir = os.path.dirname(__file__)
winner = neat.main(fitness=eval_fitness, gen_size=300, pop_size=50, verbose=True)

# Show output of the most fit genome against training data.
# print('\nBest genome:\n{!s}'.format(winner))
# print('\nOutput:')
winner_net = neat.generate_network(winner)


inputs = []
for t in range(NUM_TESTS):
    seq = ''.join(random.choice(STR_INPUT) for _ in range(STR_LEN))
    inputs.append(seq)

for seq in inputs:
    out = ''
    for i in range(len(seq)-1):
        inputs = [int(seq[i]), int(seq[i+1])]
        output = winner_net(inputs)
        # print(output[0], sep='')
        if output[0] > 0.5:
            out += 'J'
        else:
            out += seq[i]
    out += seq[i+1]
    seq = seq.replace('0', '_').replace('1', '-')
    out = out.replace('0', '_').replace('1', '-')
    print("_input: {}\noutput: {}\n".format(seq, out))

# print('Most fit individuals in each species:')
import pprint
# pprint.pprint(pop.species)

# for s in pop.species:
    # print(s.ID, ': ', [x.fitness for x in s.members])
# for k, v in pop.statistics.generation_statistics:
#     print('{:>4} : {}'.format(k, max(v)))
