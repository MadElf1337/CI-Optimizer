import random
import numpy as np
import matplotlib.pyplot as plt


# Matyas function specifics
range_matyas = 20
matyas_upper = 10
matyas_lower = -10

# Getting the input from the user
cand = []
var = []

print("Enter the number of candidates: ")
candidates = int(input())


print("Enter the number of variables: ")
variables = 2

test_array = np.empty([candidates, variables])

print("Generating random values for variables in the specified range")

for k in range(0, candidates):
    for h in range(0, variables):
        test_array[k][h] = round(random.uniform(matyas_lower, matyas_upper), 3)

print("Enter the reduction factor: ")
reduction_factor = round(float(input()), 2)


def matyas_function() -> list:
    list_fvalues = []

    for i in range(candidates):
        list_fvalues.append(
            0.26 * (test_array[i][0] ** 2 + test_array[i][1] ** 2)
            - 0.48 * (test_array[i][0] * test_array[i][1])
        )

    return list_fvalues


def prob_calc(list_fvalues) -> list:
    temp_num = np.array(list_fvalues)
    list_pvalues = []

    for i in range(0, candidates):
        list_pvalues.append(((1 / list_fvalues[i]) / sum(np.reciprocal(temp_num))))

    return list_pvalues


def roulette_prob():
    list_roulette = []
    for t in range(0, candidates):
        list_roulette.append(round(random.uniform(0, 1), 2))
    # print(list_roulette)
    return list_roulette


def roulette(list_pvalues) -> list:
    list_cumulative = [sum(list_pvalues[0:x:1]) for x in range(0, candidates + 1)]

    return list_cumulative


def attempt(list_cumulative, list_roulette):
    global range_matyas
    aux = []
    arr = np.empty([candidates, variables])

    for i in list_roulette:
        for j in range(0, len(list_cumulative) - 1):
            if list_cumulative[j] == 0:
                aux.append(0)
            elif list_cumulative[j] < i <= list_cumulative[j + 1]:
                aux.append(j)

    for lint in range(0, candidates):
        p = aux[lint]
        f = test_array[p]
        arr[lint] = f

    sampling_range = range_matyas * reduction_factor

    new_upper_bound = round(sampling_range / 2, 3)
    new_lower_bound = round(-(sampling_range / 2), 3)

    range_matyas = new_upper_bound - new_lower_bound

    sampling_intervals = np.empty([candidates, variables], dtype=tuple)
    for q in range(candidates):
        for x in range(variables):
            temp3 = arr[q][x] + new_upper_bound
            temp4 = arr[q][x] + new_lower_bound
            if temp3 > matyas_upper:
                temp3 = round(matyas_upper, 3)
            if temp4 < matyas_lower:
                temp4 = round(matyas_lower, 3)
            sampling_intervals[q][x] = (temp4, temp3)
            test_array[q][x] = round(random.uniform(temp3, temp4), 3)


if __name__ == "__main__":
    x_graph = []
    y_graph = []
    iterations = 201  # Number of iterations which are to be observed

    for i in range(1, iterations):
        x_graph.append(i)
        a = matyas_function()
        y_graph.append(a)
        c = prob_calc(a)
        d = roulette_prob()
        e = roulette(c)
        attempt(e, d)
    plt.plot(x_graph, y_graph)
    plt.xlabel("Iterations")
    plt.ylabel("function values")
    plt.show()
