from src.ObjectiveSpace import *
import pandas as pd
import os


if __name__ == '__main__':

    model_names = ['NGCF', 'ItemKNN', 'UserKNN', 'BPRMF', 'NeuMF', 'LightGCN']
    datasets = ['amazon_books', 'amazon_music', 'movielens_1m']

    for dataset in datasets:
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
                    obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2}, model_name)
                    non_dominated = obj.get_nondominated()
                    non_dominated_norm = obj.get_nondominated_norm()
                    non_dominated_hp = obj.get_nondominated_per_hp()
                    non_dominated_hp_norm = obj.get_nondominated_per_hp(norm=True)
                    if save:
                        for k, v in non_dominated_hp.items():
                            for i, j in v.items():
                                j.to_csv(
                                    f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_{k}={i}_not_dominated.csv'.replace('$','.'),
                                    sep=',', index=False)
                        non_dominated.to_csv(
                            f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_not_dominated.csv', sep=',',
                            index=False)
                        for k, v in non_dominated_hp_norm.items():
                            for i, j in v.items():
                                j.to_csv(
                                    f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_{k}={i}_not_dominated_norm.csv'.replace('$','.'),
                                    sep=',', index=False)
                        non_dominated_norm.to_csv(
                            f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_not_dominated_norm.csv', sep=',',
                            index=False)

                    dominated_hp = obj.get_dominated_per_hp()
                    dominated_hp_norm = obj.get_dominated_per_hp(norm=True)
                    if save:
                        for k, v in dominated_hp.items():
                            for i, j in v.items():
                                j.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_{k}={i}_dominated.csv'.replace('$','.'),
                                         sep=',', index=False)
                        for k, v in dominated_hp_norm.items():
                            for i, j in v.items():
                                j.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_{k}={i}_dominated_norm.csv'.replace('$','.'),
                                         sep=',', index=False)

                    std, mean = obj.get_statistics()
                    print(f'STD: {std}')
                    print(f'MEAN: {mean}')
                    stat_hp = obj.get_statistics_per_hp()
                    for k, v in stat_hp.items():
                        for i, j in v.items():
                            print(f'{k}: {i}')
                            print(f'STD: {j[0]}, MEAN: {j[1]}')
                    print(f'HP: {stat_hp}')
