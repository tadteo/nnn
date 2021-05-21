#!/usr/bin/env python3
import config
from graph import Edge,Node,Graph
from sklearn.metrics import classification_report, f1_score
import random

class Individual():
    def __init__(self):
        self.G = None
        self.score = -1

def generate_initial_population(pop_size=config.POPULATION_SIZE):
    population = []
    for i in range(pop_size):
        ind = Individual()
        ind.G = Graph(input_size=config.INPUT_SIZE,output_size=config.OUTPUT_SIZE)
        # print(f" Generating {i} individual, graph_size:{len(G.nodes)}")

        population.append(ind)
    return population

def generate_next_population(parent_1, parent_2, pop_size=config.POPULATION_SIZE):
    print("Generating next population")
    population = [parent_1,parent_2]
    # print(f"The copy of the best id is {id(population[0][0])}")
    if(pop_size>2):
        for i in range(pop_size-2): #### REMEMBER ADD -2 when adding parents to population
            
            ind = Individual()
            ind.G = Graph.fromParents(parent_1.G,parent_2.G, input_size=config.INPUT_SIZE,output_size=config.OUTPUT_SIZE)
            # print(f" Generating {i} individual, graph:{G}")
            # print(f"Generating son {i} , {len(ind.G.nodes)}")
            population.append(ind)
    return population

def test_individual(individual,batch):
    prediction = []
    # counter = 0
    batch_dimension= 200
    minibatch_X = []
    minibatch_Y = []
    for i in range(batch_dimension):
        n= random.randint(0,len(batch['X'])-1)
        minibatch_X.append(batch['X'][n])
        minibatch_Y.append(batch['Y'][n])

    for x in minibatch_X:
        # print(f"Testing input {counter}")
        # counter+=1
        # print(x)
        x= x.flatten()
        for i in x:
            i = i/255
        # print(x)
        output_node,max_output_value= individual.G.execute(x)
        # print(f"Max output node: {max_output_node}")

        prediction.append(output_node)
        
        # print("Evaluate model")
        # print(prediction)
    individual.score = f1_score(minibatch_Y, prediction, average='weighted')
    # print(individual.score)
                    
