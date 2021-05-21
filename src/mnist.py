#!/usr/bin/env python3

from concurrent.futures import thread
from itertools import count
from mimetypes import init
import random
from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
import multiprocessing as mp
import copy
import config
import utils
from evolution import generate_initial_population,generate_next_population,test_individual
from graph import Edge,Node,Graph

import threading
import time
import multiprocessing as mp
import gc

import matplotlib.pyplot as plt


def main():
    print("Executing main")
    print("Generating population")
    # pool = mp.Pool(mp.cpu_count()-2)

    population = generate_initial_population(config.POPULATION_SIZE)

    # print(population)
    print("Importing dataset")
    
    
    #loading
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    
    #shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_Y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_Y.shape))
    
    

    batch = {"X": train_X,
             "Y": train_Y}
    # print(batch)

    # Create figure for plotting
    plt.axis([0, 10000, 0, 1])

    counter=0
    while(config.NUMBER_OF_GENERATIONS>0):
        counter +=1
        print(f"Testing population of generation {counter}\n")
        counter=0

        processes = []


        for i in population:
            # print(f"\rTesting individual n {counter}")
            # print(f"Printing individual {counter},\n {p[0]}")
            # counter += 1

            process = mp.Process(target=test_individual, args=(i,batch))
            process.start()
            processes.append(process)
            # test_individual(i,batch)
            # print(i.score, len(i.G.nodes))
            # pool.map(test_individual, population)

        # Wait for all threads to complete
        for p in processes:
            p.join()

        #select the best two individuals
        best, second_best = utils.select_best_two(population)
        
        x, y = counter,best.score
        plt.scatter(x,y)
        plt.pause(0.05)


        # print("The best is ",mx,best_value_index)
        # print("The second best is",second_best_value,second_best_value_index)
        # for i in p[1]:
        #     print(i)
        test_individual(best,batch)
        print(f"\tBest individial {type(best)} f1 score: {best.score}, number of nodes: {len(best.G.nodes)}")
        test_individual(second_best,batch)
        print(f"\tSecond best individial {type(best)} f1 score: {second_best.score}, number of nodes: {len(second_best.G.nodes)}")

        # print(f"Best individial f1 score: {test_individual(population[0])}")
        # print(f"Second best individial f1 score: {test_individual(population[1])}")
        if(best.score>=0.9):
            break
        
        # population= generate_next_population(best, second_best, 3)
        population= generate_next_population(best, second_best, config.POPULATION_SIZE)
        gc.collect()
        # print(type(population),type(population[0]),type(population[0].nodes))

    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(test_X[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()    # print(iris)


if __name__ == '__main__':
    main()
    