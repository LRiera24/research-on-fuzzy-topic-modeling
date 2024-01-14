from process_jsons import process_jsons
from scatter import plot_scatter
from heatmap import plot_heatmap
from pwordcloud import plot_wordclouds
from bars import plot_bar_chart
import os
import numpy as np

GET = 0
DATA = 0
MATRIX = 1

def experiment1(processed_jsons):
    matrices = []
    for idx, pjson in enumerate(processed_jsons):
        data = pjson[DATA]
        plot_scatter(pjson[MATRIX], data, idx)
        matrices.append(pjson[MATRIX])
        # for i in range(len(pjson[DATA])):
        #     name = data[i]['corpus_name']
        #     plot_wordclouds(pjson[DATA][i]['clusters'], data[i]['words_tfidf'], f'{name}{i}')
    print(matrices)
    plot_heatmap(matrices)

def experiment2(processed_jsons):
    names = []
    mean_sims = []
    for pjson in processed_jsons:
        data = pjson[DATA]
        for i in range(len(data)):
            sim = data[i]['defs_sim']
            plot_bar_chart(data[i]['top_words'], sim)
            mean_sims.append(np.mean(sim))
        names.append(data['corpus_name'])
    plot_bar_chart(names, mean_sims, False)
    
def experiment3(processed_jsons):
    for index, pjson in enumerate(processed_jsons):
        data = pjson[DATA]
        for i in range(len(data)):
            name = data[i]['corpus_name']
            top_words = data[i]['top_words']
            tags = []
            topics = []
            for index, tag in enumerate(data[i]['estimated_tags']):
                if tag not in tags:
                    tags.append(tag)
                    topics.append(top_words[index])
            plot_wordclouds(topics, data[i]['words_tfidf'], f'{name}{index}', tags)
    pass

test_folder = os.path.abspath('tests')

# region fase1
# test_folder1 = test_folder + f'/20newsgroups/run69_20_fase1'
# test_folder2 = test_folder + f'/Brown/run14_20_fase1'
# test_folder3 = test_folder + f'/Reuters/run11_20_fase1'

# test_folder1 = test_folder + f'/20newsgroups/run53_100_fase1'
# test_folder2 = test_folder + f'/Brown/run12_100_fase1'
# test_folder3 = test_folder + f'/Reuters/run5_100_fase1'

test_folder1 = test_folder + f'/20newsgroups/run30_500_fase1'
test_folder2 = test_folder + f'/Brown/run7_500_fase1'
test_folder3 = test_folder + f'/Reuters/run3_500_fase1'
# endregion

# region fase2-wsd
# endregion

# region fase2-tagging
# test_folder += f'/20newsgroups/run41_0_fase2'
# test_folder += f'/20newsgroups/run36_0_fase2'
# test_folder += f'/20newsgroups/run34_0_fase2'
# test_folder += f'/20newsgroups/run35_0_fase2'

# test_folder += f'/Brown/run8_0_fase2'
# test_folder += f'/Brown/run9_0_fase2'
# test_folder += f'/Brown/run10_0_fase2'

# test_folder += f'/Reuters/run7_0_fase2'
# test_folder += f'/Reuters/run9_0_fase2'
# test_folder += f'/Reuters/run10_0_fase2'
# endregion

processed_jsons = []

processed_jsons.append((process_jsons(test_folder1, 13, 5)))
processed_jsons.append((process_jsons(test_folder2, 6, 5)))
processed_jsons.append((process_jsons(test_folder3, 5, 5)))
experiment1(processed_jsons)

# processed_jsons.append((process_jsons(test_folder, 13, 5, True)))
# processed_jsons.append((process_jsons(test_folder, 6, 5, True)))
# processed_jsons.append((process_jsons(test_folder, 5, 5, True)))
# experiment3(processed_jsons)