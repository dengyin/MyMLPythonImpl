from tqdm import tqdm


def kfold_mean(df_train, df_test, target, columns, kf_split):
    mean_of_target = df_train[target].mean()

    for fold_, (trn_idx, val_idx) in tqdm(enumerate(kf_split)):
        tr_x = df_train.iloc[trn_idx, :]
        vl_x = df_train.iloc[val_idx, :]

        for col in columns:
            df_train.loc[vl_x.index, f'{col}_enc_by_{target}_mean'] = vl_x[col].map(
                tr_x.groupby(col)[target].mean())

    for col in columns:
        df_train[f'{col}_enc_by_{target}_mean'].fillna(mean_of_target, inplace=True)

        df_test[f'{col}_enc_by_{target}_mean'] = df_test[col].map(
            df_train.groupby(col)[f'{col}_enc_by_{target}_mean'].mean())

        df_test[f'{col}_enc_by_{target}_mean'].fillna(mean_of_target, inplace=True)
    return df_train, df_test
