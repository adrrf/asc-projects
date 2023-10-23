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
    print("enter dimensions: ")
    dimension = int(input())
    return population_size, number_of_generations, neighborhood_size, dimension

def generate_cf6_limits(dimension):
    limits = []
    for i in range(dimension):
        if i == 0:
            limits.append((0, 1))
        else:
            limits.append((-2, 2))
    return limits
    
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

def generate_initial_population(population, dimension, limits):
    individuals = []
    for i in range(population):
        individual = []
        for j in range(dimension):
            inferior_limit, superior_limit = limits[j]
            individual.append(random.uniform(inferior_limit, superior_limit))
        individuals.append(individual)
    return individuals
def objetive1(individual):
    x_1 = individual[0]
    sum = 0
    n = len(individual)
    J = [j for j in range(2, n+1)]
    for j in J:
        if (j % 2) == 1:
            y_j = individual[j-1] - 0.8 * x_1 * math.cos(6 * math.pi * x_1 + (j*math.pi/ n))
            sum += y_j * y_j
    return x_1 + sum

def objetive2(individual):
    x_1 = individual[0]
    sum = 0
    n = len(individual)   
    J = [j for j in range(2, n+1)]
    for j in J:
        if (j % 2) == 0:
            y_j = individual[j-1] - 0.8 * x_1 * math.sin(6 * math.pi * x_1 + (j*math.pi/ n))
            sum += y_j * y_j
    return ((1 - x_1)**2) + sum

def restriction1(individual):
    x_1 = individual[0]
    x_2 = individual[1]
    sin_part = 0.8 * x_1 * math.sin(6 * math.pi * x_1 + (2 * math.pi / len(individual)))
    sgn_part = math.copysign(1, (0.5 * (1 - x_1) - (1 - x_1) ** 2))
    sqrt_part = math.sqrt(abs((0.5 * (1 - x_1) - (1 - x_1) ** 2)))
    return x_2 - sin_part - sgn_part * sqrt_part

def restriction2(individual):
    x_1 = individual[0]
    x_4 = individual[3]
    sin_part = 0.8 * x_1 * math.sin(6 * math.pi * x_1 + (4 * math.pi / len(individual)))
    sgn_part = math.copysign(1, (0.25 * math.sqrt(1 - x_1) - 0.5 * (1 - x_1)))
    sqrt_part = math.sqrt(abs((0.25 * math.sqrt(1 - x_1) - 0.5 * (1 - x_1))))
    return x_4 - sin_part - sgn_part * sqrt_part

def evaluate(population):
    evaluated_population = {}
    for individual in population:
        evaluated_population[tuple(individual)] =((objetive1(individual), objetive2(individual)))

    return evaluated_population

def constrain(population):
    constrained_population = {}
    for individual in population:
        r1 = restriction1(individual)
        r2 = restriction2(individual)
        constrained_population[tuple(individual)] = 0
        if r1 < 0:
            constrained_population[tuple(individual)] += r1
        if r2 < 0:
            constrained_population[tuple(individual)] += r2
    return constrained_population

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

def gaussian_mutation(individual, SIG, limits, p):
    for i in range(len(individual)):
        inf_limit, sup_limit = limits[i]
        sigma = (sup_limit - inf_limit) / SIG
        if random.uniform(0, 1) <= (1 / p):
            individual[i] = individual[i] + random.gauss(0, sigma)
    return individual

def reproduction(individual, population, b_i, cr, limits, dimension):    
    index1, index2, index3 = random.sample(b_i, 3)
    parent1 = population[index1]
    parent2 = population[index2]
    parent3 = population[index3]
    mutated = mutation(parent1, parent2, parent3, f=0.5)
    child = crossing(individual, mutated, cr)
    child = gaussian_mutation(child, 20, limits, dimension)
    for i in range(len(child)):
        inf_limit, sup_limit = limits[i]
        if child[i] < inf_limit:
            child[i] = inf_limit*2 - child[i]
        elif child[i] > sup_limit:
            child[i] = sup_limit*2 - child[i]
    return child
    
def tchebycheff_formulation(evaluation, z, weights):
    return max([weights[0] * abs(evaluation[0] - z[0]), weights[1] * abs(evaluation[1] - z[1])])

def neighborhood_actualization(subproblem, child, child_evaluation, child_restrictions, population, b_i, weights, z, evaluations, constraints):
    for i in range(len(b_i)):
        index = b_i[i]
        individual = population[index]
        f1, f2 = evaluations.get(tuple(individual))
        cv = constraints.get(tuple(individual))
        if (cv == 0) and (child_restrictions == 0):
            child_tc = tchebycheff_formulation(child_evaluation, z, weights[index])
            individual_tc = tchebycheff_formulation((f1, f2), z, weights[index])
            if child_tc <= individual_tc:
                population[index] = child
                break
        elif (cv < 0) and (child_restrictions == 0):
            population[index] = child
            break
        elif (cv < 0) and (child_restrictions < 0):
            if child_restrictions > cv:
                population[index] = child
                break
    return population

def export_all_pop(allpop, gen, pop):
    with open("stats/allpop"+ str(pop) + "g" + str(gen) +".out", "w") as file:
        for f1,f2 in allpop:
            file.write(str(f1) + "\t" + str(f2) + "\n")  

def export_last_gen(population, evaluations, cv):
    with open("stats/result.txt", "w") as file:
        for individual in population:
            f1, f2 = evaluations.get(tuple(individual))
            r1 = restriction1(individual)
            r2 = restriction2(individual)
            file.write(str(f1) + "\t" + str(f2) + "\t" + str(r1) + "\t" + str(r2) + "\t" + str(individual) + "\n")
    #open the pareto front data
    with open("stats/cf6/PF.dat", "r") as file:
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
    POPULATION_SIZE, NUMBER_OF_GENERATIONS, NEIGHBORHOOD_SIZE, DIMENSION = ask_for_input()
    print("|--------------------------------------------------|")
    print("| population size: " + str(POPULATION_SIZE))
    print("| number of generations: " + str(NUMBER_OF_GENERATIONS))
    print("| neighborhood size: " + str(NEIGHBORHOOD_SIZE))
    print("| dimension: " + str(DIMENSION))
    print("|--------------------------------------------------|")
    print("| generating limits...")
    LIMITS = generate_cf6_limits(DIMENSION)
    print("| generating weights vector...")
    WEIGHTS_VECTOR = generate_weights_vector(POPULATION_SIZE)
    print("| generating euclidean distance matrix...")
    DISTANCE_MATRIX = euclidean_distance_matrix(WEIGHTS_VECTOR)
    print("| generating neighborhood...")
    NEIGHBORHOOD = generate_neighborhood(POPULATION_SIZE, DISTANCE_MATRIX, NEIGHBORHOOD_SIZE)
    print("| generating initial population...")
    INITIAL_POPULATION = generate_initial_population(POPULATION_SIZE, DIMENSION, LIMITS)
    print("| evaluating initial population...")
    EVALUATED_POPULATION = evaluate(INITIAL_POPULATION)
    CONSTRAINTS_POPULATION = constrain(INITIAL_POPULATION)
    print("| getting constraints...")
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
            child = reproduction(individual, INITIAL_POPULATION, b_i, 0.5, LIMITS, DIMENSION)
            if child != None:
                child_evaluation = (objetive1(child), objetive2(child))
                r1 = restriction1(child)
                r2 = restriction2(child)
                child_restrictions = 0
                if r1 < 0:
                    child_restrictions += r1
                if r2 < 0:
                    child_restrictions += r2
                EVALUATED_POPULATION[tuple(child)] = child_evaluation
                CONSTRAINTS_POPULATION[tuple(child)] = child_restrictions
                EVALUATIONS = EVALUATIONS + 1
                if r1 >= 0 and r2 >= 0:
                    Z = compare_z(Z, child_evaluation)
                INITIAL_POPULATION = neighborhood_actualization(j, child, child_evaluation, child_restrictions, INITIAL_POPULATION, b_i, WEIGHTS_VECTOR, Z, EVALUATED_POPULATION, CONSTRAINTS_POPULATION)
                ALL_POP.append(child_evaluation)
    print("|--------------------------------------------------|")
    print("| total evaluations:", str(EVALUATIONS))
    print("| exporting results...")
    print("|--------------------------------------------------|")
    #export_all_pop(ALL_POP, NUMBER_OF_GENERATIONS, POPULATION_SIZE)
    export_last_gen(INITIAL_POPULATION, EVALUATED_POPULATION, CONSTRAINTS_POPULATION)
