#!/usr/bin/env python3

from re import L
import config
import random
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
            setattr(result, k, copy.copy(v)) # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.
        return result

class Node():

    def __init__(self,id,type):
        self.id = id #for each type id goes to zero to type size
        self.type = type #can be input, middle, output
        self.potential = 0.0
        self.input_signal = 0.0
        self.trigger=config.TRIGGER 
        self.neighbours = []
        self.neuron_output = 1.5

    def calculate_output(self):
        #first decrease the previous potential
        if (self.input_signal > 0):
            self.potential += self.input_signal
            if self.type == "middle":
                if self.potential>config.TRIGGER:
                    self.propagate_signal()
        else:
            self.potential = max(0, self.potential-config.DECAY)

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
    def __init__(self, input_size=config.INPUT_SIZE,output_size=config.OUTPUT_SIZE, middle_size=None, nodes=[]):
        self.input_size = input_size
        if middle_size == None:
            self.middle_size = random.randint(config.MIN_INITIAL_NODES,config.MAX_INITIAL_NODES)
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
                            if random.random()>config.NEW_EDGE_PROB:
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
    def fromParents(cls, parent_1, parent_2, input_size=config.INPUT_SIZE,output_size=config.OUTPUT_SIZE):
        """generate graph from parents"""
        
        parent_base = None
        other_parent = None
        if random.choice((1,2)) == 1:
            nodes = copy.deepcopy(parent_1.nodes)
            parent_base = parent_1
            other_parent = parent_2
            middle_size = parent_1.middle_size
        else:
            nodes = copy.deepcopy(parent_2.nodes)
            parent_base = parent_2
            other_parent = parent_1
            middle_size = parent_2.middle_size

        total_size= input_size+middle_size+output_size
        # print(f"Parent base: {parent_base.total_size}")
        # # for n in nodes:
        # #     print(type(n))
        # #     for e in n.neighbours:
        # #         print(f"\t{type(e)}")
        # #         print(f"\t\t{type(e.node)}")
        # #crossover
        for i in range(len(nodes)):
            if nodes[i].type == "input" or nodes[i].type == "middle": #not needed crossover on output layer
                if random.random() < config.CROSSOVER_PROB:
                    if i < len(other_parent.nodes):
                        nodes[i] = other_parent.nodes[i]
                        #check if there are incompatible edges --> edges that brings to nodes in middle layer not existing
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
        for _ in range(middle_size,config.MAX_MIDDLE_SIZE):
            if random.random() < config.NEW_NODE_PROB:
                if middle_size < config.MAX_MIDDLE_SIZE:
                    middle_size +=1
                    node = Node(middle_size,"middle")
                    #generating edges
                    for j in nodes:
                        #There are not edges arriving to input layers
                        if j.type != "input":
                            if random.random()>config.NEW_EDGE_PROB:
                                weight = random.random() # appending random weight at the start
                                node.neighbours.append(Edge(j,weight))
                    nodes.append(node)
                         
        # #output layer ok from parent
        
        # #modify weights
        for i in nodes:
            if i.type == "input" or i.type == "middle": # output layer does not have weights
                for w in i.neighbours:
                    #modify edges with certain probability
                    if random.random() < config.MUTATION_PROB_EDGES:
                        if random.randint(0,1) == 0: #delete edge
                            i.neighbours.remove(w)
                        else: #create edge
                            weight = random.random() # random weight at the start
                            n = random.randint(input_size,total_size-1) #choosing a random node excluding input nodes
                            i.neighbours.append(Edge(nodes[n],weight))
                    #modify weights with certain probability    
                    if random.random() < config.MUTATION_PROB_WEIGHTS:
                        w.weight += random.uniform(-config.MAX_SEVERITY_OF_MUTATION, +config.MAX_SEVERITY_OF_MUTATION)
        # print(nodes)
        return cls(input_size = input_size, 
                    middle_size = middle_size,
                    output_size = output_size,
                    nodes = nodes)

    def execute(self,input):
        #putting all potential to 0 for new input
        for i in self.nodes:
            i.potential = 0

        time_left = config.MAX_TIME
        while time_left>0:
            # print(time_left)
            time_left -=1
            
            input_values = config.INPUT_SIZE-1
            for n in self.nodes:
                if n.type == "input":
                    n.neuron_output = input[input_values]
                    n.input_signal = input[input_values]
                    input_values-=1
                n.calculate_output()
                    # print(i)

            #putting the input signal from this iteration to 0
            for i in self.nodes:
                i.input_signal = 0
        
        

        #saving output
        max_output_value=-1
        max_output_node=None
        for i in self.nodes:
            if i.type == "output":
                if( i.potential > max_output_value):
                    max_output_value= i.potential
                    max_output_node= i.id
        return max_output_node,max_output_value

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
