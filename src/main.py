#!/usr/bin/env python3

from concurrent.futures import thread
from itertools import count
from mimetypes import init
import random
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, f1_score
import numpy as np
import multiprocessing as mp
import copy
import config
import utils
from evolution import generate_initial_population,generate_next_population,test_individual
from graph import Edge,Node,Graph

import threading
import time
import multiprocessing
import datetime as dt
import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt




def main():
    print("Executing main")
    print("Generating population")
    # pool = mp.Pool(mp.cpu_count()-2)

    population = generate_initial_population(config.POPULATION_SIZE)

    # print(population)
    print("Importing dataset")
    iris = load_iris()
    # print(iris)
    # print(iris.data[:-15])
    # print(iris.data[-15:])
    generation_counter = config.NUMBER_OF_GENERATIONS

    batch = {"X": iris.data,
             "Y": iris.target}
    # print(batch)

    # Create figure for plotting
    plt.axis([0, 10000, 0, 1])

    
    counter=0
    while(counter<generation_counter):
        counter +=1
        print(f"Testing population of generation {config.NUMBER_OF_GENERATIONS-generation_counter}")
        

        processes = []

        for i in population:
            process = mp.Process(target=test_individual, args=(i,batch))
            process.start()
            processes.append(process)

        # Wait for all threads to complete
        for p in processes:
            p.join()

        best, second_best = utils.select_best_two(population)
        
        x, y = counter,best.score
        plt.scatter(x,y)
        plt.pause(0.05)

        test_individual(best,batch)
        print(f"Best individial {type(best)} f1 score: {best.score}, number of nodes: {len(best.G.nodes)}")
        test_individual(second_best,batch)
        print(f"Second best individial {type(best)} f1 score: {second_best.score}, number of nodes: {len(second_best.G.nodes)}")

        # print(f"Best individial f1 score: {test_individual(population[0])}")
        # print(f"Second best individial f1 score: {test_individual(population[1])}")
        if(best.score>=0.9):
            break
        
        population= generate_next_population(best, second_best, config.POPULATION_SIZE)
        # print(type(population),type(population[0]),type(population[0].nodes))


if __name__ == '__main__':
    main()

    