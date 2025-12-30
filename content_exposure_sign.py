from src.ObjectiveSpace import *
import pandas as pd
import os
from itertools import combinations
from utils.statistical_significance_test import statistical_significance


if __name__ == '__main__':

    lookup_hp = {
        'LightGCN': ['factors', 'n_layers', 'lr'],
        'NGCF': ['factors', 'n_layers', 'lr'],
        'ItemKNN': ['nn', 'sim'],
        'UserKNN': ['nn', 'sim'],
        'BPRMF': ['f', 'lr'],
        'NeuMF': ['factors', 'lr']
    }

    scenario = 'content_delivery'
    model_names = ['NGCF', 'ItemKNN', 'UserKNN', 'BPRMF', 'NeuMF', 'LightGCN']
    datasets = ['amazon_books', 'amazon_music', 'movielens_1m']
    families = [('ItemKNN', 'UserKNN'),('NGCF', 'LightGCN'),('BPRMF', 'NeuMF')]
    norm = 2
    distances = {}
    distances_hp = {}
    for dataset in datasets:
        print(dataset)
        distances[dataset] = {}
        distances_hp[dataset] = {}
        champ_global = []
        champ_hp = []
        print(f"---{dataset}---")
        for model_name in model_names:
            dir = os.listdir(f'data/{dataset}/{model_name}')
            obj1 = 'nDCG'
            opt1 = 'max'
            obj2 = 'APLT'
            opt2 = 'max'
            save = False
            plot = False

            for element in dir:
                if 'cutoff_10' in element:
                    model = pd.read_csv(f'data/{dataset}/{model_name}/{element}', sep='\t')
                    obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2}, model_name, norm)
                    distances[dataset][model_name] = obj.get_distances()
                    distances_hp[dataset][model_name] = obj.get_distances_per_hp()

        for m1, m2 in combinations(distances[dataset].keys(), 2):
            print(m1, m2)
            data1 = np.array(distances[dataset][m1])
            data2 = np.array(distances[dataset][m2])
            if len(data1) < 2 or len(data2) < 2:
                continue

            p_mu, p_sigma = statistical_significance(data1, data2)
            print(f"{m1} vs {m2} TTEST - μ: p-value = {p_mu:.4f}")
            # nprint(f"{m1} vs {m2} LEVENE - σ: p-value = {p_sigma:.4f}")

        for model_name in model_names:
            couples = list(combinations(lookup_hp[model_name], 2))
            for couple in couples:
                v1 = list(distances_hp[dataset][model_name][couple[0]].keys())
                v2 = list(distances_hp[dataset][model_name][couple[1]].keys())
                for k1 in v1:
                    for k2 in v2:
                        data1 = list(distances_hp[dataset][model_name][couple[0]][k1].values())
                        data2 = list(distances_hp[dataset][model_name][couple[1]][k2].values())
                        p_mu, p_sigma = statistical_significance(data1, data2)
                        print(f"{model_name}, {couple[0]} = {k1} vs {couple[1]} = {k2}, TTEST - μ: p-value = {p_mu:.4f}")






        for family in families:
            for hp in lookup_hp[family[0]]:
                for k, value in distances_hp[dataset][family[0]][hp].items():
                    try:
                        data1 = list(distances_hp[dataset][family[0]][hp][k].values())
                        data2 = list(distances_hp[dataset][family[1]][hp][k].values())
                    except KeyError:
                        hp = 'f'
                        data1 = list(distances_hp[dataset][family[0]][hp][k].values())
                        hp = 'factors'
                        data2 = list(distances_hp[dataset][family[1]][hp][k].values())
                    p_mu, p_sigma = statistical_significance(data1, data2)
                    print(f"{family[0]} vs {family[1]}, {hp}:{k} TTEST - μ: p-value = {p_mu:.4f}")
                    # print(f"{m1} vs {m2} LEVENE - σ: p-value = {p_sigma:.4f}")

        #for family in families:






        pass