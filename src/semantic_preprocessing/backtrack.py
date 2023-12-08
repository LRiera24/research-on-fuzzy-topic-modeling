import numpy as np
from nltk.corpus import wordnet


def average_distance_between_synsets(synsets):
    if len(synsets) < 2:
        return 0.0  # No pairs to compare

    total_distance = 0.0
    pair_count = 0

    for i in range(len(synsets)):
        for j in range(i + 1, len(synsets)):
            distance = synsets[i].path_similarity(synsets[j])
            if distance is not None:
                total_distance += distance
                pair_count += 1

    if pair_count == 0:
        return 0.0  # No valid pairs
    
    return total_distance / pair_count


def backtrack_solution(words):

    # boolean array to mark selected candidates
    w = [-1 for i in range(len(words))]
    print(f'w: {w}')

    synsets = [wordnet.synsets(word) for word in words]
    print(f'syn: {synsets}')

    s = [[False for _ in inner_list] for inner_list in synsets]
    print(f's: {s}')

    # list of best assigment of candidates and position
    best_solution = []

    # list of solutions
    answer = []

    _backtrack_solution(s, w, synsets,
                        best_solution, answer)

    print(len(best_solution))
    value = max(answer)
    print(value)
    solution = best_solution[answer.index(value)]
    print(solution)
    for index, value in enumerate(solution):
        print(synsets[index][value].definition())

    return solution.tolist(), value


def _backtrack_solution(s, w, synsets, best_solution, answer):

    # base case
    if np.sum(1 for word in w if word > -1) == len(w):
        answer.append(average_distance_between_synsets(
            [synsets[i][w[i]] for i in range(len(w))]))
        best_solution.append(np.copy(w))
        return

    # in each iteration one candidate is assign to a position
    for i in range(len(w)):
        # print("FOR TRAN")
        for j in range(len(s[i])):
            # print('FOR TREN')
            if w[i] == -1:
                s[i][j] = True
                w[i] = j
                # print(w)
                _backtrack_solution(s, w, synsets, best_solution, answer)
                s[i][j] = False
                w[i] = -1


context = ['homework', 'pencil', 'school', 'write', 'notebook']
backtrack_solution(context)
