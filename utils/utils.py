import pandas as pd


def df_to_latex_rows(df: pd.DataFrame) -> str:

    lines = []

    header = " & ".join(df.columns) + " \\\\"
    lines.append(header)

    # Righe con i valori
    for _, row in df.iterrows():
        formatted = []
        for val in row:
            if isinstance(val, (int, float)):
                formatted.append(str(int(round(val))))
            else:
                formatted.append(str(val))
        line = " & ".join(formatted) + " \\\\"
        lines.append(line)

    return "\n".join(lines)


def arrange_results(df, mode):
    if mode == 'overall':
        result = df[['rank_borda', 'model', 'borda_score']].sort_values('rank_borda')
    elif mode == 'mean':
        result = df[['rank_borda_mean', 'model', 'borda_mean']].sort_values('rank_borda_mean')
    else:
        result = df[['rank_borda_std', 'model', 'borda_std']].sort_values('rank_borda_std')
    return result

def borda_count(champ, scenario, dataset, case, norm, weight_mean, weight_std):
    df = pd.DataFrame(champ, columns=['model', 'mean', 'std'])
    n = len(df)
    df['rank_mean'] = df['mean'].rank(ascending=True, method='min')
    df['rank_std'] = df['std'].rank(ascending=True, method='min')
    df['borda_mean'] = (n - df['rank_mean']) * weight_mean
    df['borda_std'] = (n - df['rank_std']) * weight_std
    df['rank_borda_mean'] = df['borda_mean'].rank(ascending=False, method='min')
    df['rank_borda_std'] = df['borda_std'].rank(ascending=False, method='min')
    df['borda_score'] = df['borda_mean'] + df['borda_std']
    df['rank_borda'] = df['borda_score'].rank(ascending=False, method='min')
    borda_overall = arrange_results(df, 'overall')
    borda_mean = arrange_results(df, 'mean')
    borda_std = arrange_results(df, 'std')
    borda_overall.to_csv(f'borda_counts/{scenario}_{case}_{dataset}_{norm}_overall.tsv', sep='\t', index=False)
    borda_mean.to_csv(f'borda_counts/{scenario}_{case}_{dataset}_{norm}_mean.tsv', sep='\t', index=False)
    borda_std.to_csv(f'borda_counts/{scenario}_{case}_{dataset}_{norm}_std.tsv', sep='\t', index=False)
    # return df
