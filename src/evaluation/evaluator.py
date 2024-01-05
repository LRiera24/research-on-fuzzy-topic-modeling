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
    for pjson in processed_jsons:
        plot_scatter(pjson[MATRIX])
        matrices.append(pjson[MATRIX])
        for i in range(len(pjson[DATA])):
            plot_wordclouds(pjson[DATA][i]['clusters'])
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
    for pjson in processed_jsons:
        data = pjson[DATA]
        for i in range(len(data)):
            plot_wordclouds(data[i]['top_words'], data[i]['estimated_tags'])
    pass

test_folder = os.path.abspath('tests')
test_folder += f'/20newsgroups/run29_1000_individual'

processed_jsons = []
processed_jsons.append((process_jsons(test_folder, 13, 5)))
# print(processed_jsons)
# experiment1(processed_jsons)
experiment3(processed_jsons)