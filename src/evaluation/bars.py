import matplotlib.pyplot as plt
import numpy as np

# Combinando las dos funciones anteriores en una sola para generar ambas gráficas

def graficar_desempeno_wsd_combinado(resultados):
    """
    Grafica el desempeño de los algoritmos WSD en una sola función, produciendo dos figuras:
    una para el desempeño por corpus y otra para el desempeño general.

    :param resultados: Un diccionario que contiene los valores de similitud semántica.
    """
    # Nombres de los corpus y algoritmos
    corpus_nombres = list(resultados.keys())
    algoritmos_nombres = list(resultados[corpus_nombres[0]].keys())
    
    # Número de corpus y algoritmos
    n_corpus = len(corpus_nombres)
    n_algoritmos = len(algoritmos_nombres)

    # Valores para cada algoritmo en cada corpus
    valores = [[np.mean(resultados[corpus][algoritmo]) for algoritmo in algoritmos_nombres] for corpus in corpus_nombres]

    # Crear figura para el desempeño por corpus
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Posiciones de las barras y tonos de azul
    bar_width = 0.35
    indices = np.arange(n_corpus)
    colores_azules = ['royalblue', 'skyblue']

    # Crear barras para cada algoritmo en el primer gráfico
    for i in range(n_algoritmos):
        ax1.bar(indices + i * bar_width, [valores[j][i] for j in range(n_corpus)], bar_width, color=colores_azules[i], label=algoritmos_nombres[i])

    # Añadir etiquetas y título al primer gráfico
    ax1.set_xlabel('Corpus')
    ax1.set_ylabel('Average semantic similarity')
    ax1.set_title('Performance of WSD algorithms per Corpus')
    ax1.set_xticks(indices + bar_width / 2)
    ax1.set_xticklabels(corpus_nombres)
    ax1.legend()

    plt.savefig('wsd_corpus')

    # Calcular la media de similitud semántica para cada algoritmo en todos los corpus para el segundo gráfico
    media_valores = {'Extended Lesk': 484,
                    'Genetic Algorithm': 18114}
    # for algoritmo in algoritmos_nombres:
    #     media_valores[algoritmo] = np.mean([np.mean(resultados[corpus][algoritmo]) for corpus in resultados])

    # Crear figura para el desempeño general
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Crear barras para el segundo gráfico
    ax2.bar(media_valores.keys(), media_valores.values(), color='purple')

    # Añadir etiquetas y título al segundo gráfico
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Average execution time (s)')
    ax2.set_title('General Performance of WSD algorithms')

    # Ajustar layout y mostrar gráficos
    plt.tight_layout()
    plt.savefig('execution_time_general')
    return fig1, fig2

resultados = {
    '20-Newsgroups': {
        'Extended Lesk': [
            0.08615072843550764, 0.07813533652515235, 0.08846135510413078, 0.08350047317757615,
            0.09756390915788192, 0.08555091565667633, 0.09242177851111245, 0.07996762090851087,
            0.07631248729855074, 0.08168364191102566, 0.08659139030921355, 0.08695177801876705,
            0.09829682022629513, 0.07537479502030761, 0.08246099067325686, 0.08554729143206301,
            0.08688394068375058, 0.08668213338114307, 0.08207067323987687, 0.08520382464965542
        ],
        'Genetic Algorithm': [
            0.09442695278574445, 0.09878740746661115, 0.11409620110170542, 0.09056244845261216,
            0.09313211619843831, 0.09709150455617636, 0.08913169160979685, 0.09264485358216144,
            0.08706196387559524, 0.10285409233478889, 0.09236748703209581, 0.10500964769993512,
            0.10714821492040347, 0.08945715709194589, 0.09743339724567479, 0.08741055471407165,
            0.10695349424793985, 0.10480402974946401, 0.10144461998404834, 0.08626470393605931
        ]
    },
    'Brown': {
        'Extended Lesk': [
            0.07569358372166882, 0.08458649960918453, 0.08847766297238113, 0.08003397100596486,
            0.07984009700144173, 0.0828461458734325, 0.08707189677202376, 0.08264701883576636,
            0.09470057835854052, 0.08281850671125904, 0.0931430334386161, 0.08499246644788662,
            0.09694876030953144, 0.08768976922005907, 0.08569574523026767
        ],
        'Genetic Algorithm': [
            0.10598669166328162, 0.09212978615514437, 0.09924895093144964, 0.09954861666072265,
            0.09504825960918224, 0.10628025543810707, 0.09384073100982596, 0.09485135215541027,
            0.09989974596256238, 0.08970277638068291, 0.09230221065627553, 0.0949649081484774,
            0.09225415589403835, 0.09773454329596962, 0.08974635531711944
        ]
    },
    'Reuters': {
        'Extended Lesk': [
            0.08621985229693144, 0.08627535157492541, 0.08627535157492541, 0.08627535157492541, 
            0.08627535157492541, 0.08627535157492541, 0.08627535157492541, 0.0878370033934706, 
            0.09319664480881784, 0.08627535157492541, 0.08627535157492541, 0.09268972771631676, 
            0.08627535157492541, 0.08627535157492541, 0.08627535157492541, 0.08627535157492541, 
            0.08627535157492541, 0.08627535157492541, 0.08627535157492541, 0.08627535157492541, 
            0.08627535157492541
        ],
        'Genetic Algorithm': [
            0.0962885907186944, 0.09190770540744823, 0.09936580541993373, 0.10008542949093555,
            0.0951285279535772, 0.09665140793764788, 0.09585875174226206, 0.09596018380738218,
            0.09468921787289525, 0.09525196655087159, 0.09455306713677633, 0.09760374400610833,
            0.09460914277359136, 0.10292266948222699, 0.09707826131253985, 0.0949456499700151,
            0.09630924500469895, 0.09403886125706251, 0.09473939514488422, 0.09434714795702281,
            0.09496496532474108
        ]
        }
        }

# Generar y mostrar ambas gráficas
fig1, fig2 = graficar_desempeno_wsd_combinado(resultados)
plt.show()
