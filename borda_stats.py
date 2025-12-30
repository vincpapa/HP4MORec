import pandas as pd
import os
import re
from utils.utils import df_to_latex_rows

def aggregate_borda_scores(folder_path='borda_counts', type_filter=None):

    pattern = re.compile(
        r'^(content_delivery|content_exposure)_(global|hp)_(amazon_books|amazon_music|movielens_1m)_(1|2|inf)_(overall|std|mean)\.tsv$'
    )

    files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]

    grouped_files = {}

    for filename in files:
        match = pattern.match(filename)
        if not match:
            continue
        scenario, granularity, dataset, norm, type_ = match.groups()
        if type_filter and type_ != type_filter:
            continue
        key = (scenario, granularity, norm, type_)
        if key not in grouped_files:
            grouped_files[key] = {}
        grouped_files[key][dataset] = os.path.join(folder_path, filename)

    results = {}

    for key, dataset_files in grouped_files.items():
        scenario, granularity, norm, type_ = key

        score_col = {
            'overall': 'borda_score',
            'mean': 'borda_mean',
            'std': 'borda_std'
        }[type_]

        dfs = []

        for dataset, filepath in dataset_files.items():
            df = pd.read_csv(filepath, sep='\t')

            if 'model' not in df.columns or score_col not in df.columns:
                print(f"⚠️  Skipping {filepath}: missing '{score_col}' column.")
                continue

            df = df[['model', score_col]].rename(columns={score_col: f'borda_score_{dataset}'})
            dfs.append(df)

        if not dfs:
            print(f"⚠️  No valid datasets for {key}, skipping.")
            continue

        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='model', how='outer')

        merged_df['borda_score_tot'] = merged_df.filter(like='borda_score_').sum(axis=1, skipna=True)

        merged_df['rank_borda'] = merged_df['borda_score_tot'].rank(ascending=False, method='min')

        expected_cols = ['borda_score_amazon_books', 'borda_score_amazon_music', 'borda_score_movielens_1m']
        for c in expected_cols:
            if c not in merged_df.columns:
                merged_df[c] = pd.NA

        merged_df = merged_df[['rank_borda', 'model'] + expected_cols + ['borda_score_tot']].sort_values('rank_borda')

        results[key] = merged_df

    return results

if __name__ == '__main__':
    results = aggregate_borda_scores('borda_counts')

    scenario = 'content_exposure'
    granularity = 'hp'
    norm = 'inf'
    type_ = 'overall'

    df_result = results.get((scenario, granularity, norm, type_))
    if df_result is not None:
        if granularity == 'hp':
            df_result['hyperparameter'] = df_result['model'].map(lambda x: x.split('_')[1])
            df_result['value'] = df_result['model'].map(lambda x: x.split('_')[2])
            df_result['model'] = df_result['model'].map(lambda x: x.split('_')[0])
            print(df_to_latex_rows(df_result[['rank_borda','model','hyperparameter','value','borda_score_amazon_books','borda_score_amazon_music','borda_score_movielens_1m','borda_score_tot']]))
        else:
            print(df_result)
    else:
        print("Nessun risultato per questa combinazione.")