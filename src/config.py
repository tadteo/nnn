#!/usr/bin/env python3
#
# ### Iris
#
TRIGGER = 2.45 #minimum potential to activate the neuron
DECAY = 0.25   #decay of the potential for each timestep
MAX_TIME = 20

MIN_INITIAL_NODES = 20 #20
MAX_INITIAL_NODES = 40 #50
MAX_MIDDLE_SIZE = 80

POPULATION_SIZE = 10 #60

MUTATION_PROB_EDGES= 0.015 #probability of adding or removing an edge
MUTATION_PROB_WEIGHTS= 0.2 #probability of changing the or removing an edge
MAX_SEVERITY_OF_MUTATION=0.3 #maximum severity of a mutation allowed

CROSSOVER_PROB = 0.05
NEW_NODE_PROB = 0.1

#minimum probability to append a new branch
NEW_EDGE_PROB=0.6

INPUT_SIZE=4
OUTPUT_SIZE=3
NUMBER_OF_GENERATIONS = 10000



#
### MNIST
#

# TRIGGER = 2.45 #minimum potential to activate the neuron
# DECAY = 0.25   #decay of the potential for each timestep
# MAX_TIME = 20

# MIN_INITIAL_NODES = 20 #20
# MAX_INITIAL_NODES = 40 #50
# MAX_MIDDLE_SIZE = 80

# POPULATION_SIZE = 60

# MUTATION_PROB_EDGES= 0.005 #probability of adding or removing an edge
# MUTATION_PROB_WEIGHTS= 0.02 #probability of changing the or removing an edge
# MAX_SEVERITY_OF_MUTATION=0.2 #maximum severity of a mutation allowed

# CROSSOVER_PROB = 0.005
# NEW_NODE_PROB = 0.01

# #minimum probability to append a new branch
# NEW_EDGE_PROB=0.6

# INPUT_SIZE= 28*28
# OUTPUT_SIZE=10
# NUMBER_OF_GENERATIONS = 1000
