#!/usr/bin/env python3


TRIGGER = 2.45 #minimum potential to activate the neuron
DECAY = 0.25   #decay of the potential for each timestep
MAX_TIME = 20

MIN_INITIAL_NODES = 20 #20
MAX_INITIAL_NODES = 40 #50
MAX_MIDDLE_SIZE = 80

POPULATION_SIZE = 60

MUTATION_PROB_EDGES= 0.005 #probability of adding or removing an edge
MUTATION_PROB_WEIGHTS= 0.02 #probability of changing the or removing an edge
MAX_SEVERITY_OF_MUTATION=0.2 #maximum severity of a mutation allowed

CROSSOVER_PROB = 0.005
NEW_NODE_PROB = 0.01

#minimum probability to append a new branch
NEW_EDGE_PROB=0.6

INPUT_SIZE=4
OUTPUT_SIZE=3
NUMBER_OF_GENERATIONS = 1000

from concurrent.futures import thread
from itertools import count
from mimetypes import init
import random
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, f1_score
import numpy as np
import multiprocessing as mp
import copy


class Edge():
    def __init__(self, node, weight):
        self.node=node
        self.weight=weight

    def __repr__(self):
        return f"Connected with Node {self.node.id}-{self.node.type} id:{id(self.node)}, weight: {self.weight}, actual_input_signal: {self.node.input_signal}"

    def __str__(self):
        return f"Connected with Node {self.node.id}-{self.node.type} id:{id(self.node)}, weight: {self.weight}, actual_input_signal: {self.node.input_signal}"

    def __deepcopy__(self, memo):
        cls = self.__class__ # Extract the class of the object
        result = cls.__new__(cls) # Create a new instance of the object based on extracted class
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v,memo)) # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.
        return result

class Node():

    def __init__(self,id,type) -> None:
        self.id = id #for each type id goes to zero to type size
        self.type = type #can be input, middle, output
        self.potential = 0.0
        self.input_signal = 0.0
        self.trigger=TRIGGER 
        self.neighbours = []
        self.neuron_output = 1.5

    def calculate_output(self):
        #first decrease the previous potential
        if (self.input_signal > 0):
            self.potential += self.input_signal
            if self.type == "middle":
                if self.potential>TRIGGER:
                    self.propagate_signal()
        else:
            self.potential = max(0, self.potential-DECAY)

    def propagate_signal(self):
        for n in self.neighbours:
            n.node.input_signal += self.neuron_output*n.weight
        self.potential = 0

    def __repr__(self):
        base= f"\nNode {self.id} {id(self)}, type: {self.type}, actual_potential: {self.potential}, actual_input: {self.input_signal}, trigger: {self.trigger}\n"
        for i in self.neighbours:
            base += f"{i}, "
        return base

    def __str__(self):
        base= f"\nNode {self.id} {id(self)}, type: {self.type}, actual_potential: {self.potential}, actual_input: {self.input_signal}, trigger: {self.trigger}\n"
        for i in self.neighbours:
            base += f"\t{i}\n"
        return base

    def __deepcopy__(self, memo):
        cls = self.__class__ # Extract the class of the object
        result = cls.__new__(cls) # Create a new instance of the object based on extracted class
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo)) # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.
        return result

class Graph():

    #Constructur for initial population
    def __init__(self, input_size=INPUT_SIZE,output_size=OUTPUT_SIZE, middle_size=None, nodes=[]):
        self.input_size = input_size
        if middle_size == None:
            self.middle_size = random.randint(MIN_INITIAL_NODES,MAX_INITIAL_NODES)
        else:
            self.middle_size = middle_size
        self.output_size = output_size
        self.total_size= input_size+self.middle_size+output_size
        
        if nodes==[]:
            self.nodes = []
            #generate nodes
            id = 0 #output_size+size+input_size
            #generate input layer
            for n in range(self.input_size):
                node=Node(id,"input")
                self.nodes.append(node)
                id = id+1

            id = 0 
            #generate central graph
            for n in range(self.middle_size):
                node = Node(id,"middle")
                self.nodes.append(node)
                id = id+1 

            id = 0  
            #generate output layer
            for n in range(self.output_size):
                self.nodes.append(Node(id,"output"))
                id = id+1
            
            #generate weights
            for i in self.nodes:
                #There are not edges coming out from output nodes
                if i.type != "output":
                    for j in self.nodes:
                        #There are not edges arriving to input layers
                        if j.type != "input":
                            if random.random()>NEW_EDGE_PROB:
                                weight = random.random() # appending random weight at the start
                                i.neighbours.append(Edge(j,weight))

            # for n in self.nodes:
            #     print(type(n))
            #     for e in n.neighbours:
            #         print(f"\t{type(e)}")
            #         print(f"\t\t{type(e.node)}")

        else:
            self.nodes = nodes

    #generate graph from parents
    #possible mutations are:
    # increase number of node
    # weight change
    # neighbours change
    @classmethod
    def fromParents(cls, parent_1, parent_2, input_size=INPUT_SIZE,output_size=OUTPUT_SIZE):
        """generate graph from parents"""
        # middle_size = random.choice((parent_1.middle_size,parent_2.middle_size))
        middle_size = parent_1.middle_size
        total_size= input_size+middle_size+output_size
        parent_base = None
        other_parent = None
        if middle_size == parent_1.middle_size:
            nodes = copy.deepcopy(parent_1.nodes)
            parent_base = parent_1
            other_parent = parent_2
        else:
            nodes = copy.deepcopy(parent_2.nodes)
            parent_base = parent_1
            other_parent = parent_2
        # print(f"Parent base: {parent_base.total_size}")
        # # for n in nodes:
        # #     print(type(n))
        # #     for e in n.neighbours:
        # #         print(f"\t{type(e)}")
        # #         print(f"\t\t{type(e.node)}")
        # #crossover
        for i in range(len(nodes)):
            if nodes[i].type == "input" or nodes[i].type == "middle": #not needed crossover on output layer
                if random.random() < CROSSOVER_PROB:
                    if i < len(other_parent.nodes):
                        nodes[i] = other_parent.nodes[i]
                        #check if there are uncompatible edges --> edges that brings to nodes in middle layer not existing
                        for e in nodes[i].neighbours:
                            # print(type(e))
                            if e.node.type == "middle" and e.node.id > middle_size-1:
                                nodes[i].neighbours.remove(e)

                            #modify the node to the references of the actual graph
                            for x in nodes:
                                if (e.node.id == x.id and e.node.type == x.type ):
                                    e.node = x
                            # e.node = next(x for x in nodes if (e.node.id == x.id and e.node.type == x.type )) 

       
        #modify or generate nodes
        #input layer number of nodes ok from parent
        #mutate central graph number of nodes
        for _ in range(middle_size,MAX_MIDDLE_SIZE):
            if random.random() < NEW_NODE_PROB:
                if middle_size < MAX_MIDDLE_SIZE:
                    middle_size +=1
                    node = Node(middle_size,"middle")
                    #generating edges
                    for j in nodes:
                        #There are not edges arriving to input layers
                        if j.type != "input":
                            if random.random()>NEW_EDGE_PROB:
                                weight = random.random() # appending random weight at the start
                                node.neighbours.append(Edge(j,weight))
                    nodes.append(node)
                         
        # #output layer ok from parent
        
        # #modify weights
        for i in nodes:
            if i.type == "input" or i.type == "middle": # output layer does not have weights
                for w in i.neighbours:
                    #modify edges with certain probability
                    if random.random() < MUTATION_PROB_EDGES:
                        if random.randint(0,1) == 0: #delete edge
                            i.neighbours.remove(w)
                        else: #create edge
                            weight = random.random() # random weight at the start
                            n = random.randint(input_size,total_size-1) #choosing a random node excluding input nodes
                            i.neighbours.append(Edge(nodes[n],weight))
                    #modify weights with certain probability    
                    if random.random() < MUTATION_PROB_WEIGHTS:
                        w.weight += random.uniform(-MAX_SEVERITY_OF_MUTATION, +MAX_SEVERITY_OF_MUTATION)
        # print(nodes)
        return cls(input_size = input_size, 
                    middle_size = middle_size,
                    output_size = output_size,
                    nodes = nodes)

    def __repr__(self):
        base= f"\nGraph:\n\tTotal_size={self.total_size}, input_size: {self.input_size}, middle_size: {self.middle_size}, output_size: {self.output_size}\n"
        for i in self.nodes:
            base += f"\t{i}, "
        return base

    def __str__(self):
        base= f"\nGraph:\n\tTotal_size={self.total_size}, input_size: {self.input_size}, middle_size: {self.middle_size}, output_size: {self.output_size}\n"
        for i in self.nodes:
            base += f"\t{i}, "
        return base
        
    def __deepcopy__(self, memo):
        cls = self.__class__ # Extract the class of the object
        result = cls.__new__(cls) # Create a new instance of the object based on extracted class
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo)) # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.
        return result
        

def generate_initial_population(pop_size=POPULATION_SIZE):
    population = []
    for i in range(pop_size):
        G = Graph(input_size=INPUT_SIZE,output_size=OUTPUT_SIZE)
        # print(f" Generating {i} individual, graph_size:{len(G.nodes)}")
        population.append([G,-1])
    print("len of population:",len(population))
    return population

def generate_next_population(parent_1, parent_2, pop_size=POPULATION_SIZE):
    # population = [[parent_1,-1]]
    population = [[parent_1,-1],[parent_2,-1]]
    # print(f"The copy of the best id is {id(population[0][0])}")
    for i in range(pop_size-2): #### REMEMBER ADD -2 when adding parents to population
        G = Graph.fromParents(parent_1,parent_2, input_size=INPUT_SIZE,output_size=OUTPUT_SIZE)
        # print(f" Generating {i} individual, graph:{G}")
        population.append([G,-1])
    print("len of population:",len(population))
    return population

import threading
import time
import multiprocessing

def main():
    print("Executing main")
    print("Generating population")
    # pool = mp.Pool(mp.cpu_count()-2)

    population = generate_initial_population(POPULATION_SIZE)

    # print(population)
    print("Importing dataset")
    iris = load_iris()
    # print(iris)
    # print(iris.data[:-15])
    # print(iris.data[-15:])
    generation_counter = NUMBER_OF_GENERATIONS

    while(generation_counter>0):
        generation_counter -=1
        print(f"Testing population of generation {NUMBER_OF_GENERATIONS-generation_counter}")
        counter=0

        # threadLock = threading.Lock()
        threads = []

        # class myThread (threading.Thread):
        #     def __init__(self, threadID, counter, p ):
        #         threading.Thread.__init__(self)
        #         self.threadID = threadID
        #         self.counter = counter
        #         self.p = p

        #     def run(self):
        #         # print ("Starting " + self.name)
        #         # Get lock to synchronize threads
        #         threadLock.acquire()
        #         test_individual(self.p)
        #         # Free lock to release next thread
        #         threadLock.release()


        for p in population:
            # print(f"\rTesting individual n {counter}")
            # print(f"Printing individual {counter},\n {p[0]}")
            counter += 1

            def test_individual(p):
                
                batch = iris.data
                prediction = []
                for b in batch:
                    time_left = MAX_TIME
                    while time_left>0:
                        time_left -=1
                        
                        input_values = INPUT_SIZE-1
                        for i in p[0].nodes:
                            if i.type == "input":
                                i.input_signal = b[input_values]
                                input_values-=1
                                for j in i.neighbours:
                                    j.node.input_signal += i.input_signal*j.weight
                                # print(i)

                        for n in p[0].nodes:
                            n.calculate_output()
                            # print(n)
                        
                        #putting the input signal from this iteration to 0
                        for i in p[0].nodes:
                            i.input_signal = 0
                    
                    

                    #saving output
                    max_output_value=-1
                    max_output_node=None
                    for i in p[0].nodes:
                        if i.type == "output":
                            if( i.potential > max_output_value):
                                max_output_value= i.potential
                                max_output_node= i.id
                    # print(f"Max output node: {max_output_node}")

                    # print(f"length of graph: {len(p)}")
                    for i in range(OUTPUT_SIZE):
                        # print(i,p.nodes[i])
                        if max_output_node == p[0].nodes[i].id:
                            prediction.append(i)
                    
                    #putting all potential to 0 for new input
                    for i in p[0].nodes:
                            i.potential = 0
                    
                    
                # print("Evaluate model")
                # print(prediction)
                prediction = np.array(prediction)
                # print(prediction)
                report = (f1_score(iris.target, prediction, average='weighted'))
                # print(type(report))
                p[1] = report
                return report

            # t=myThread(counter,counter,p)
            # t = multiprocessing.Process(target=test_individual, args=(p,))
            # t.start()
            # threads.append(t)
            print(test_individual(p))
            # pool.map(test_individual, population)
        
        # for i in population[1]:
        #     print(f"results: {i}")

        # Wait for all threads to complete
        # for t in threads:
        #     t.join()

        #select the best two individuals
        mx=max(population[0][1],population[1][1])
        second_best_value=min(population[0][1],population[1][1])
        n =len(population)
        for i in range(2,n):
            if population[i][1]>mx:
                second_best_value=mx
                mx=population[i][1]
            elif population[i][1]>second_best_value and \
                mx != population[i][1]:
                second_best_value=population[i][1]

        for i in range(len(population)):
            if population[i][1] == mx:
                best_value_index= i
            if population[i][1] == second_best_value:
                second_best_value_index= i
        best = population[best_value_index][0]
        second_best = population[second_best_value_index][0]

        print("The best is ",mx,best_value_index)
        print("The second best is",second_best_value,second_best_value_index)
        # for i in p[1]:
        #     print(i)

        print(f"Best individial {type(best)} f1 score: {test_individual(population[best_value_index])}")
        print(f"Second best individial {type(best)} f1 score: {test_individual(population[second_best_value_index])}")

        # print(f"Best individial f1 score: {test_individual(population[0])}")
        # print(f"Second best individial f1 score: {test_individual(population[1])}")
        if(mx>=0.9):
            break

        population= generate_next_population(best, second_best, POPULATION_SIZE)
        # print(type(population),type(population[0]),type(population[0].nodes))


if __name__ == '__main__':
    main()
    