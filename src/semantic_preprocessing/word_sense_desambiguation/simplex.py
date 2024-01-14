from pulp import LpVariable, LpProblem, lpSum, value, LpMinimize, LpMaximize
from nltk.corpus import wordnet


def create_optimization_problem(N, K, distance_function, synsets):
    # Create LP problem
    problem = LpProblem(name="ItemSelection", sense=LpMaximize)

    # Create decision variables
    x = {(i, j): LpVariable(name=f"x_{i}_{j}", cat="Binary")
         for i in range(1, N + 1) for j in range(1, K[i - 1] + 1)}

    # Create auxiliary variables
    y = {(i, j, i_prime, j_prime): LpVariable(name=f"y_{i}_{j}_{i_prime}_{j_prime}", cat="Binary")
         for i in range(1, N + 1)
         for j in range(1, K[i - 1] + 1)
         for i_prime in range(1, N + 1)
         if i_prime != i
         for j_prime in range(1, K[i_prime - 1] + 1)}

    calculated_distances = []

    # Create objective function
    objective = lpSum(distance_function(i, j, i_prime, j_prime, synsets, calculated_distances) * y[i, j, i_prime, j_prime]
                      for i in range(1, N + 1)
                      for j in range(1, K[i - 1] + 1)
                      for i_prime in range(1, N + 1)
                      if i_prime != i
                      for j_prime in range(1, K[i_prime - 1] + 1))
    print("!!!!!!!!!!!!!!!!!!!!", calculated_distances)
    problem += objective / len(calculated_distances)

    # Create constraints
    for i in range(1, N + 1):
        problem += lpSum(x[i, j] for j in range(1, K[i - 1] + 1)) == 1

    for i in range(1, N + 1):
        for j in range(1, K[i - 1] + 1):
            for i_prime in range(1, N + 1):
                if i_prime != i:
                    for j_prime in range(1, K[i_prime - 1] + 1):
                        problem += y[i, j, i_prime, j_prime] >= x[i,
                                                                  j] + x[i_prime, j_prime] - 1
                        problem += y[i, j, i_prime, j_prime] <= x[i, j]
                        problem += y[i, j, i_prime,
                                     j_prime] <= x[i_prime, j_prime]

    return problem, x


def solve_optimization_problem(problem):
    problem.solve()
    return problem

def custom_distance_function(i, j, i_prime, j_prime, synsets, calculated_distances):
    synset1 = synsets[i-1][j-1]
    synset2 = synsets[i_prime-1][j_prime-1]
    if (synset1, synset2) in calculated_distances or (synset2, synset1) in calculated_distances or synset1.pos() == 'v' or synset2.pos() == 'v':
        return 0
    print(synset1, synset2)
    calculated_distances.append((synset1, synset2))
    return synset1.path_similarity(synset2)


def print_solution(x, N, K, synsets):
    res = []
    for i in range(1, N + 1):
        chosen_item = [j for j in range(
            1, K[i-1] + 1) if value(x[i, j]) == 1][0]
        chosen_item = synsets[i-1][chosen_item-1]
        res.append(chosen_item)
        print(
            f"Chosen definition: {chosen_item} {chosen_item.definition()}")
            # f"Word: {context[i-1]} ---> Chosen definition: {chosen_item} {chosen_item.definition()}")
    return res

def simplex_sol(synsets):
    N = len(synsets)  # Number of elements
    # Number of items for each element
    K = [len(word_synset) for word_synset in synsets]
    print(N, K)

    problem, x = create_optimization_problem(
    N, K, custom_distance_function, synsets)
    solve_optimization_problem(problem)

    print("Objective value:", value(problem.objective), '\n')

    r = print_solution(x, N, K, synsets)

    return r

# Example usage
# context = ['dog', 'cat', 'rabbit']
# context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']
# context = ['birthday', 'party', 'gift', 'music', 'candles', 'wish']
# context = ['computer', 'program', 'development', 'web', 'application', 'data']
# context = ['school', 'class', 'homework', 'student', 'book', 'knowledge', 'learn', 'teach', 'library', 'computer']

# context = ['people', 'game', 'year', 'time', 'know', 'thing', 'day', 'point']
# synsets = [wordnet.synsets(word) for word in context if wordnet.synsets(word)]
# print(synsets)
# r = simplex_sol(synsets)
# print(r)

# N = len(context)  # Number of elements
# # Number of items for each element
# K = [len(word_synset) for word_synset in synsets]
# print(N, K)

# problem, x = create_optimization_problem(
#     N, K, custom_distance_function, synsets)
# solve_optimization_problem(problem)

# print("Objective value:", value(problem.objective), '\n')

# r = print_solution(x, N, K, context, synsets)
# print(r)