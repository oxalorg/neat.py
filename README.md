neat.py
=======

My small personal implementation of
[NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
(NeuroEvolution of Augmenting Topologies).

Work in progress.

## Usage:

The only thing needed by `neat.py` is a fitness function.

This fitness function recieves a list of individuals (i.e. a
population) and it is expected to alter the fitness of those
individuals.

Each individual is nothing but a dictionary (hashmap) and
includes a `fitness` key.

Check the `xor.py` file for an example. You can test it out by
simply running `python3 xor.py`.
