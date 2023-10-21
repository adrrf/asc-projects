import random
import math
import matplotlib.pyplot as plt

def ask_for_input():
    print("enter the population size: ")
    population_size = int(input())
    print("enter the number of generations: ")
    number_of_generations = int(input())
    print("enter the neighborhood size: ")
    neighborhood_size = int(input())
    print("enter the search space inferior limit: ")
    inferior_limit = int(input())
    print("enter the search space superior limit: ")
    superior_limit = int(input())
    return population_size, number_of_generations, neighborhood_size, inferior_limit, superior_limit

def generate_weights_vector(popultion_size):
    weights_vector = []
    step = 1 / (popultion_size - 1)
    for i in range(popultion_size):
        weights = []
        weights.append(i * step)
        weights.append(1 - (i * step))
        weights_vector.append(weights)
    return weights_vector

def euclidean_distance(vector1, vector2):
    distance = 0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i]) ** 2
    return distance ** 0.5

def euclidean_distance_matrix(weights_vector):
    distance_matrix = {}
    for i in range(len(weights_vector)):
        for j in range(len(weights_vector)):
            if not ((i,j) in distance_matrix) or not ((j,i) in distance_matrix):
                distance = euclidean_distance(weights_vector[i], weights_vector[j])
                distance_matrix[(i, j)] = distance
    return distance_matrix

def generate_neighborhood(population, distance_matrix, neighborhood):
    b_i = []
    for i in range(population):
        lambda_distance = [(key2, value) for (key1, key2), value in distance_matrix.items() if key1 == i]
        lambda_distance.sort(key=lambda x: x[1])
        neighbor_lambda = [index for index, _ in lambda_distance[:neighborhood]]
        b_i.append(neighbor_lambda)
    return b_i

def generate_initial_population(population, dimension, inferior_limit, superior_limit):
    individuals = []
    for i in range(population):
        individual = []
        for j in range(dimension):
            individual.append(random.uniform(inferior_limit, superior_limit))
        individuals.append(individual)
    return individuals

def objetive1(individual):
    return individual[0]

def objetive2(individual):
    f1 = objetive1(individual)
    g = 1 + 9 * (sum(individual[1:]) / (len(individual) - 1))
    h = 1 - ((f1 / g) ** 0.5) - (f1 / g * (math.sin(10 * math.pi * f1)))
    return g * h

def evaluate(population):
    evaluated_population = {}
    for individual in population:
        evaluated_population[tuple(individual)] =((objetive1(individual), objetive2(individual)))
    return evaluated_population

def get_z(evaluations):
    return [min([x[0] for x in evaluations.values()]), min([x[1] for x in evaluations.values()])]

def compare_z(z_previous, z_current):
    res = z_previous
    if z_current[0] < z_previous[0]:
        res[0] = z_current[0]
    if z_current[1] < z_previous[1]:
        res[1] = z_current[1]
    return res

def crossing(individual, mutated, cr):
    child = []
    delta = random.randint(0, len(individual) - 1)
    for i in range(len(individual)):
        if random.uniform(0, 1) <= cr or i == delta:
            child.append(mutated[i])
        else:
            child.append(individual[i])
    return child

def mutation(parent1, parent2, parent3, f):
    child = []
    for i in range(len(parent1)):
        child.append(parent1[i] + f * (parent2[i] - parent3[i]))
    return child

def gaussian_mutation(individual, SIG, inf_limit, sup_limit, p):
    sigma = (sup_limit - inf_limit) / SIG
    for i in range(len(individual)):
        if random.uniform(0, 1) <= (1 / p):
            individual[i] = individual[i] + random.gauss(0, sigma)
    return individual

def reproduction(individual, population, b_i, cr, inf_limit, sup_limit):    
    index1, index2, index3 = random.sample(b_i, 3)
    parent1 = population[index1]
    parent2 = population[index2]
    parent3 = population[index3]
    mutated = mutation(parent1, parent2, parent3, f=0.5)
    child = crossing(individual, mutated, cr)
    child = gaussian_mutation(child, 20, inf_limit, sup_limit, 30)
    for i in range(len(child)):
        if child[i] < inf_limit:
            child[i] = inf_limit*2 - child[i]
        elif child[i] > sup_limit:
            child[i] = sup_limit*2 - child[i]
    return child
    
def tchebycheff_formulation(evaluation, z, weights):
    return max([weights[0] * abs(evaluation[0] - z[0]), weights[1] * abs(evaluation[1] - z[1])])

def neighborhood_actualization(child, child_evaluation, population, b_i, weights, z, evaluations):
    for i in range(len(b_i)):
        index = b_i[i]
        individual = population[index]
        f1, f2 = evaluations.get(tuple(individual))
        child_tc = tchebycheff_formulation(child_evaluation, z, weights[index])
        individual_tc = tchebycheff_formulation((f1, f2), z, weights[index])
        if child_tc <= individual_tc:
            population[index] = child
    return population

def export_all_pop(allpop, gen, pop):
    with open("stats/allpop"+ str(pop) + "g" + str(gen) +".out", "w") as file:
        for f1,f2 in allpop:
            file.write(str(f1) + "\t" + str(f2) + "\n")  


def export_last_gen(population, evaluations):
    with open("stats/result.txt", "w") as file:
        for individual in population:
            f1, f2 = evaluations.get(tuple(individual))
            file.write(str(f1) + "\t" + str(f2) + "\n")  
    #open the pareto front data
    with open("stats/PF.dat", "r") as file:
        data = file.readlines()
    #plot the pareto front
    x = [float(line.split("\t")[0]) for line in data]
    y = [float(line.split("\t")[1]) for line in data]
    plt.scatter(x, y, color="red")
    #plot the results 
    x = [evaluations.get(tuple(individual))[0] for individual in population]
    y = [evaluations.get(tuple(individual))[1] for individual in population]
    plt.scatter(x, y)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()

if __name__ == "__main__":
    POPULATION_SIZE, NUMBER_OF_GENERATIONS, NEIGHBORHOOD_SIZE, INFERIOR_LIMIT, SUPERIOR_LIMIT = ask_for_input()
    print("|--------------------------------------------------|")
    print("| population size: " + str(POPULATION_SIZE))
    print("| number of generations: " + str(NUMBER_OF_GENERATIONS))
    print("| neighborhood size: " + str(NEIGHBORHOOD_SIZE))
    print("| search space inferior limit: " + str(INFERIOR_LIMIT))
    print("| search space superior limit: " + str(SUPERIOR_LIMIT))
    print("|--------------------------------------------------|")
    print("| generating weights vector...")
    WEIGHTS_VECTOR = generate_weights_vector(POPULATION_SIZE)
    print("| generating euclidean distance matrix...")
    DISTANCE_MATRIX = euclidean_distance_matrix(WEIGHTS_VECTOR)
    print("| generating neighborhood...")
    NEIGHBORHOOD = generate_neighborhood(POPULATION_SIZE, DISTANCE_MATRIX, NEIGHBORHOOD_SIZE)
    print("| generating initial population...")
    INITIAL_POPULATION = generate_initial_population(POPULATION_SIZE, 30, INFERIOR_LIMIT, SUPERIOR_LIMIT)
    print("| evaluating initial population...")
    EVALUATED_POPULATION = evaluate(INITIAL_POPULATION)
    print("| getting z...")
    Z = get_z(EVALUATED_POPULATION)
    EVALUATIONS = 0
    ALL_POP = []
    print("|--------------------------------------------------|")
    print("| starting algorithm...")
    print("|--------------------------------------------------|")
    for i in range(NUMBER_OF_GENERATIONS):
        print("| generation: " + str(i))
        print("| z: " + str(Z))
        for j in range(POPULATION_SIZE):
            individual = INITIAL_POPULATION[j]
            b_i = NEIGHBORHOOD[j]
            child = reproduction(individual, INITIAL_POPULATION, b_i, 0.5, INFERIOR_LIMIT, SUPERIOR_LIMIT)
            if child != None:
                child_evaluation = (objetive1(child), objetive2(child))
                EVALUATED_POPULATION[tuple(child)] = child_evaluation
                EVALUATIONS = EVALUATIONS + 1
                Z = compare_z(Z, child_evaluation)
                INITIAL_POPULATION = neighborhood_actualization(child, child_evaluation, INITIAL_POPULATION, b_i, WEIGHTS_VECTOR, Z, EVALUATED_POPULATION)
                ALL_POP.append(child_evaluation)
    print("|--------------------------------------------------|")
    print("| total evaluations:", str(EVALUATIONS))
    print("| exporting results...")
    print("|--------------------------------------------------|")
    export_all_pop(ALL_POP, NUMBER_OF_GENERATIONS, POPULATION_SIZE)
    export_last_gen(INITIAL_POPULATION, EVALUATED_POPULATION)
