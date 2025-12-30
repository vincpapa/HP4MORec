from src.ObjectiveSpace import *
import pandas as pd
import os
from utils.utils import borda_count


if __name__ == '__main__':

    scenario = 'content_exposure'
    model_names = ['NGCF', 'ItemKNN', 'UserKNN', 'BPRMF', 'NeuMF', 'LightGCN']
    datasets = ['amazon_books', 'amazon_music', 'movielens_1m']
    norm = np.inf


    for dataset in datasets:
        champ_global = []
        champ_hp = []
        print(f"---{dataset}---")
        for model_name in model_names:
            print(f'--- {model_name} --- {dataset} ---')
            dir = os.listdir(f'data/{dataset}/{model_name}')
            obj1 = 'nDCG'
            opt1 = 'max'
            obj2 = 'APLT'
            opt2 = 'max'
            save = False
            plot = False

            for element in dir :
                if 'cutoff_10' in element:
                    model = pd.read_csv(f'data/{dataset}/{model_name}/{element}', sep='\t')
                    obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2}, model_name, norm)
                    non_dominated = obj.get_nondominated()
                    non_dominated_norm = obj.get_nondominated_norm()
                    non_dominated_hp = obj.get_nondominated_per_hp()
                    non_dominated_hp_norm = obj.get_nondominated_per_hp(norm=True)
                    if save:
                        for k, v in non_dominated_hp.items():
                            for i, j in v.items():
                                j.to_csv(
                                    f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{k}={i}_not_dominated.csv'.replace('$','.'),
                                    sep=',', index=False)
                        non_dominated.to_csv(
                            f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_not_dominated.csv', sep=',',
                            index=False)
                        for k, v in non_dominated_hp_norm.items():
                            for i, j in v.items():
                                j.to_csv(
                                    f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{k}={i}_not_dominated_norm.csv'.replace('$','.'),
                                    sep=',', index=False)
                        non_dominated_norm.to_csv(
                            f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_not_dominated_norm.csv', sep=',',
                            index=False)

                    dominated_hp = obj.get_dominated_per_hp()
                    dominated_hp_norm = obj.get_dominated_per_hp(norm=True)
                    if save:
                        for k, v in dominated_hp.items():
                            for i, j in v.items():
                                j.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{k}={i}_dominated.csv'.replace('$','.'),
                                         sep=',', index=False)
                        for k, v in dominated_hp_norm.items():
                            for i, j in v.items():
                                j.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{k}={i}_dominated_norm.csv'.replace('$','.'),
                                         sep=',', index=False)

                    std, mean = obj.get_statistics()
                    print(f'STD: {std}')
                    print(f'MEAN: {mean}')
                    champ_global.append([model_name, mean, std])
                    stat_hp = obj.get_statistics_per_hp()
                    for k, v in stat_hp.items():
                        for i, j in v.items():
                            print(f'{k}: {i}')
                            print(f'STD: {j[0]}, MEAN: {j[1]}')
                            champ_hp.append([f'{model_name}_{k}_{i}', j[1], j[0]])
                    print(f'HP: {stat_hp}')
        borda_count(champ_global, scenario, dataset, 'global', norm, 2, 1)
        borda_count(champ_hp, scenario, dataset, 'hp', norm, 2, 1)