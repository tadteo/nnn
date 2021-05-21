#select the best two individuals
def select_best_two(population):
    if population[0].score >= population[1].score:
        mx=population[0].score
        best = population[0]
        second_best_value = population[1].score
        second_best = population[1]
    else:
        mx=population[1].score
        best = population[1]
        second_best_value = population[0].score
        second_best = population[0]

    for i in range(2,len(population)):
        if population[i].score>mx:
            second_best_value = mx
            second_best = best 
            mx=population[i].score
            best=population[i]
            
        elif population[i].score>second_best_value and mx != population[i].score:
            second_best_value=population[i].score
            second_best=population[i]
    return best, second_best
