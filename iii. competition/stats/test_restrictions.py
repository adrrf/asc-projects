# read last_pop.txt skip lines with # and get the 4 first columns separated by tabulations
# and put them in a list of lists
import math

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

def read():
    with open('last_pop.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if not line.startswith('#')]
        lines = [line.split('\t') for line in lines]
        #parse to float
        lines = [[float(x) for x in line] for line in lines]
        lines = [[line[0], line[1], line[2], line[3], line[4:8]] for line in lines]
        
        return lines
    
def compare():
    lines = read()
    for f1, f2, r1, r2, individual in lines:
        print(individual)
        print("the f1 " + str(f1))
        print("the f2 " + str(f2))
        print("the r1 " + str(r1))
        print("the r2 " + str(r2))
        print("my r1 " + str(restriction1(individual)))
        print("my r2 " + str(restriction2(individual)))
        print("my f1 " + str(objetive1(individual)))
        print("my f2 " + str(objetive2(individual)))

def constrain(individual):
    print(individual)
    print("my f1 " + str(objetive1(individual)))
    print("my f2 " + str(objetive2(individual)))
    print("my r1 " + str(restriction1(individual)))
    print("my r2 " + str(restriction2(individual)))

constrain([0.7654331852403002, 0.17857359547747842, -0.36003439984765234, -0.5239475061432948])

#compare()