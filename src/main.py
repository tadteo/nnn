#!/usr/bin/env python3

TRIGGER = 2.45 #minimum potential to activate the neuron
DECAY = 0.25   #decay of the potential for each timestep
MAX_TIME = 50

MIN_INITIAL_NODES = 20
MAX_INITIAL_NODES = 50

POPULATION_SIZE = 10

#minimum probability to append a new branch
NEW_EDGE_PROB=0.6

INPUT_SIZE=4
OUTPUT_SIZE=3

import random
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import numpy as np



class Edge():
    def __init__(self, node, weight):
        self.node=node
        self.weight=weight

    def __repr__(self):
        return f"Node {self.node.id}, weight: {self.weight}, actual_input_signal: {self.node.input_signal}"

    def __str__(self):
        return f"Node {self.node.id}, weight: {self.weight}, actual_input_signal: {self.node.input_signal}"


class Node():

    def __init__(self,id,type) -> None:
        self.id = id
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
        base= f"\nNode {self.id}, type: {self.type}, actual_potential: {self.potential}, actual_input: {self.input_signal}, trigger: {self.trigger}\n"
        for i in self.neighbours:
            base += f"{i}, "
        return base

    def __str__(self):
        base= f"\nNode {self.id}, type: {self.type}, actual_potential: {self.potential}, actual_input: {self.input_signal}, trigger: {self.trigger}\n"
        for i in self.neighbours:
            base += f"\t{i}\n"
        return base

def generate_graph(input_size=INPUT_SIZE,output_size=OUTPUT_SIZE):
    middle_size = random.randint(MIN_INITIAL_NODES,MAX_INITIAL_NODES)
    total_size= input_size+middle_size+output_size
    G = []
    #generate nodes
    id = 0 #output_size+size+input_size
    #generate input layer
    for n in range(input_size):
        node=Node(id,"input")
        G.append(node)
        id = id+1
        
    #generate central graph
    for n in range(middle_size):
        node = Node(id,"middle")
        G.append(node)
        id = id+1 
        
    #generate output layer
    for n in range(output_size):
        G.append(Node(id,"output"))
        id = id+1
    
    #generate weights
    for i in G:
        #There are not edges coming out from output nodes
        if i.type != "output":
            for j in G:
                #There are not edges arriving to input layers
                if j.type != "input":
                    if random.random()>NEW_EDGE_PROB:
                        weight = random.random() # appending random weight at the start
                        i.neighbours.append(Edge(j,weight))
    return G


    

def generate_initial_population(pop_size=POPULATION_SIZE):
    population = []
    for i in range(pop_size):
        population.append(generate_graph())
    return population

def main():
    print("Executing main")
    print("Generating population")
    population = generate_initial_population(POPULATION_SIZE)

    print(population)
    print("Importing dataset")
    iris = load_iris()
    print(iris)
    print(iris.data[:-15])
    print(iris.data[-15:])
    print("Testing population")
    counter=0
    population_record=[]
    for p in population:
        print(f"Testing population n {counter}")
        counter += 1

        batch = iris.data
        prediction = []
        for b in batch:
            time_left = MAX_TIME
            while time_left>0:
                time_left -=1
                
                input_values = INPUT_SIZE-1
                for i in p:
                    if i.type == "input":
                        i.input_signal = b[input_values]
                        input_values-=1
                        for j in i.neighbours:
                            j.node.input_signal += i.input_signal*j.weight
                        # print(i)

                for n in p:
                    n.calculate_output()
                    # print(n)
                
                #putting the input signal from this iteration to 0
                for i in p:
                    i.input_signal = 0
            
            

            #saving output
            max_output_value=-1
            max_output_node=None
            for i in p:
                if i.type == "output":
                    if( i.potential > max_output_value):
                        max_output_value= i.potential
                        max_output_node= i.id
            # print(f"Max output node: {max_output_node}")

            # print(f"length of graph: {len(p)}")
            for i in range(len(p)-1,-1,-1):
                # print(i,p[i])
                if max_output_node == p[i].id:
                    prediction.append((len(p)-i-1))
            
            #putting all potential to 0 for new input
            for i in p:
                    i.potential = 0
            
            
        print("Evaluate model")
        # print(prediction)
        prediction = np.array(prediction)
        # print(prediction)
        report = classification_report(iris.target, prediction)
        # print(report)
        population_record.append(report)

    for i in population_record:
        print(i)


if __name__ == '__main__':
    main()
    