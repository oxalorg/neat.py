import neat

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def fitness(pop):
    """
    Recieves a list of pop. Modify ONLY their
    fitness values
    """
    for g in pop:
        g['fitness'] = 4.0
        nw_activate = neat.generate_network(g)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = nw_activate(xi)
            g['fitness'] -= (output[0] - xo[0]) ** 2


neat.main(fitness=fitness, gen_size=900, pop_size=50, verbose=True, fitness_thresh=3.95)
