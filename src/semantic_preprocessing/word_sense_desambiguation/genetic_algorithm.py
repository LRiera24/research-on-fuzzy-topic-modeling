import random
# from semantic_preprocessing.word_sense_desambiguation.avg_distance import calculate_mean_similarity
from semantic_preprocessing.word_sense_desambiguation.avg_distance import calculate_mean_similarity
from math import inf
from nltk.corpus import wordnet

SYNSET = 0
FITNESS = 1

def init_population(context_synsets, pop_size):
    """
    Generate an initial population of synset sets for genetic algorithm.

    Parameters:
    - context_synsets (list): List of lists, where each inner list contains synsets for a word in the context.
    - pop_size (int): Number of individuals to create in population.

    Returns:
    - population (list): Initial population of synset sets.
    """
    population = []

    for _ in range(pop_size):
        individual = []
        for synsets_for_word in context_synsets:
            synset_mask = [0 for _ in synsets_for_word]
            selected_index = random.randint(0, len(synsets_for_word) - 1)
            synset_mask[selected_index] = 1
            individual.append(synset_mask)
        population.append(individual)
    return population


def crossover(parent_set_1, parent_set_2):
    """
    Perform crossover operation on two parent synset sets.

    Parameters:
    - parent_set_1 (list): First parent synset set.
    - parent_set_2 (list): Second parent synset set.

    Returns:
    - children (list): List containing two children synset sets.
    """
    crossover_point = random.randint(0, len(parent_set_1) - 2)
    # print(len(parent_set_1))
    # print(crossover_point)

    child_set_1 = parent_set_1.copy()
    child_set_2 = parent_set_2.copy()

    temp = child_set_1[0:crossover_point]
    child_set_1[0:crossover_point] = child_set_2[0:crossover_point]
    child_set_2[0:crossover_point] = temp

    return [child_set_1, child_set_2]


def mutate(individual):
    """
    Perform mutation operation on a list of synset sets.

    Parameters:
    - individual (list): List of synset sets to be mutated.

    Returns:
    - mutated_set (list): Mutated list of synset sets.
    """
    k = random.randint(0, len(individual))

    for _ in range(k):
        synset_set_index = random.randint(0, len(individual) - 1)
        synset_set = individual[synset_set_index]

        if len(synset_set) == 1:
            continue

        active_index = synset_set.index(1)
        synset_set[active_index] = 0

        synset_index = random.randint(0, len(synset_set) - 1)
        synset_set[synset_index] = 1

        individual[synset_set_index] = synset_set

    return individual


def evaluate(synsets_masks, synsets):
    """
    Calculate the mean similarity between the selected synsets based on the given binary masks.

    Parameters:
    - synsets_masks (list): List of binary masks representing the selected synset for each word.
    - synsets (list): List of synsets to choose from.

    Returns:
    - float: The mean similarity between the selected synsets.
    """
    chosen_synsets = []
    for i, mask in enumerate(synsets_masks):
        chosen_synsets.extend([synsets[i][j]
                              for j in range(len(mask)) if mask[j] > 0 if synsets[i][j].pos() != 'v'])
    # print(chosen_synsets)
    # if len(chosen_synsets) != len(synsets_masks):
    #     return -100

    calculated_distances = []
    sim = 0
    c = 0
    for synset1 in chosen_synsets:
        for synset2 in chosen_synsets:
            if synset1 == synset2 or (synset1, synset2) in calculated_distances or (synset2, synset1) in calculated_distances or synset1.pos() == 'v' or synset2.pos() == 'v':
                continue
            calculated_distances.append((synset1, synset2))
            sim += synset1.path_similarity(synset2)
            c += 1
    if c > 0:
        sim = sim / c
    else: 
        sim = -1
    # print('fitness', sim)
    return sim


def select_parents(population, fitness_values, selection_proportion=1):
    """
    Select the best individuals as parents based on their fitness values using roulette wheel selection.

    Parameters:
    - population (list): Current population of synset sets.
    - fitness_values (list): List of fitness values corresponding to the population.
    - selection_proportion (float): Proportion of individuals to select as parents.

    Returns:
    - best_individuals (list): List of best parent synset sets.
    """
    num_parents = int(len(population) * selection_proportion)
    parents = []

    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]

    for _ in range(num_parents):
        selected_index = roulette_wheel_selection(probabilities)
        parents.append(population[selected_index])

    return parents


def roulette_wheel_selection(probabilities) -> int:
    """
    Perform roulette wheel selection to choose an index based on given probabilities.

    Parameters:
    - probabilities (list): List of probabilities.

    Returns:
    - selected_index (int): Index selected based on roulette wheel.
    """
    spin = random.uniform(0, 1)
    cumulative_probability = 0.0

    for i, prob in enumerate(probabilities):
        cumulative_probability += prob
        if spin <= cumulative_probability:
            return i

    # Fallback: In case of numerical issues, return the last index
    return len(probabilities) - 1


def new_population(best_individuals, xover_ratio=0.6, mutation_ratio=0.4):
    """
    Generate a new population of synset sets using crossover and mutation operations.

    Parameters:
    - best_individuals (list): List of best parent synset sets.
    - xover_ratio (float): Crossover ratio.
    - mutation_ratio (float): Mutation ratio.

    Returns:
    - new_population (list): New population of synset sets.
    """
    num_crossover = int(len(best_individuals) * xover_ratio)
    # print("num_crossover", num_crossover)
    num_mutation = int(len(best_individuals) * mutation_ratio)
    # print("num_mutation", num_mutation)
    # print()

    new_population = []

    # Perform crossover
    while len(new_population) < num_crossover:
        parent_1 = random.choice(best_individuals)
        # print("parent_1", parent_1)
        parent_2 = random.choice(best_individuals)
        # print("parent_2", parent_2)
        children = crossover(parent_1, parent_2)
        # print("children", children)
        random_choice = random.choices(children + [children], k=1)[0]
        # print("random_choice", random_choice)
        if len(random_choice) == 2:
            new_population.extend(random_choice)
        else:
            new_population.append(random_choice)
        # print()

    # print("Pop after xover")
    # for ind in new_population:
    #     print(ind)
    # print("Pop size after xover", len(new_population))
    # print()

    while len(new_population) < num_crossover + num_mutation:
        parent = random.choice(best_individuals)
        # print("parent", parent)
        mutated_child = mutate(parent)
        # print("mutated_child", mutated_child)
        new_population.append(mutated_child)
        # print()

    # print("Pop after mutation")
    # for ind in new_population:
    #     print(ind)
    # print("Pop size after mutation", len(new_population))
    # print()

    return new_population


def genetic_algorithm(synsets, generations=100, pop_size=50):
    population = init_population(synsets, pop_size)
    # print(len(population))
    # for individual in population:
    #     print(individual)

    # best solution found
    best_solution = ([], -inf)  # (solution, fitness value)

    for _ in range(generations):
        fitness_values = [evaluate(individual, synsets)
                          for individual in population]

        max_fitness = max(fitness_values)
        if max_fitness > best_solution[FITNESS]:
            # print("!!!!!!!!! BETTER SOLUTION FOUND")
            # i = fitness_values.index(max_fitness)
            # print(i)
            # print(population[i])
            best_solution = (
                population[fitness_values.index(max_fitness)], max_fitness)

        best_individuals = select_parents(population, fitness_values)
        population = new_population(best_individuals)
        # print("len(population)", len(population))
        # for individual in population:
        #     print(individual)

    chosen_synsets = []
    for index, synset_mask in enumerate(best_solution[SYNSET]):
        # print(synset_mask)
        d = synset_mask.index(1)
        chosen_synsets.append(synsets[index][d])

    return chosen_synsets, best_solution[FITNESS]


# context = ['salt', 'sugar', 'pepper']
# synsets = [wordnet.synsets(word) for word in context]
# print(synsets)

# chosen, value = genetic_algorithm(synsets)
# defs = [s.definition() for s in chosen]

# for i in range(len(chosen)):
#     print(chosen[i], defs[i])

# print(value)