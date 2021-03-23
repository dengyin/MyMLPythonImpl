from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


def null_importance_feature_select(all_features, train, target_, model, r):
    target = train[target_]
    print('Start with features number:', len(all_features))
    train_x = train[all_features]
    n_splits = 2
    n_runs = 5
    imp_df = np.zeros((len(train_x.columns), n_splits * n_runs))
    np.random.seed(42)
    idx = np.arange(len(target))
    for run in range(n_runs):
        np.random.shuffle(idx)
        perm_target = target.iloc[idx]
        folds = KFold(n_splits, True, None)
        oof = np.empty(len(train_x))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(perm_target, perm_target)):
            trn_dat, trn_tgt = train_x.iloc[trn_idx, :], perm_target.iloc[trn_idx]
            val_dat, val_tgt = train_x.iloc[val_idx, :], perm_target.iloc[val_idx]
            model.fit(trn_dat, trn_tgt, eval_set=[(val_dat, val_tgt)], verbose=100)
            imp_df[:, n_splits * run + fold_] = model.feature_importances_
            oof[val_idx] = model.predict(val_dat)
    bench_imp_df = np.zeros((len(train_x.columns), n_splits * n_runs))
    for run in range(n_runs):
        np.random.shuffle(idx)
        perm_target = target.iloc[idx]
        perm_data = train_x.iloc[idx]
        folds = KFold(n_splits, True, None)
        oof = np.empty(len(train_x))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(perm_target, perm_target)):
            trn_dat, trn_tgt = perm_data.iloc[trn_idx], perm_target.iloc[trn_idx]
            val_dat, val_tgt = perm_data.iloc[val_idx], perm_target.iloc[val_idx]
            model.fit(trn_dat, trn_tgt, eval_set=[(val_dat, val_tgt)], verbose=100)
            bench_imp_df[:, n_splits * run + fold_] = model.feature_importances_
            oof[val_idx] = model.predict(val_dat)
    bench_mean = bench_imp_df.mean(axis=1)
    perm_mean = imp_df.mean(axis=1)
    values = []
    for i, f in enumerate(train_x.columns):
        values.append((f, bench_mean[i], perm_mean[i], bench_mean[i] / perm_mean[i]))
    values = sorted(values, key=lambda x: x[3])
    target_fe = pd.DataFrame(values)
    target_fe.columns = ['feature', 'b', 'p', 'r']
    target_fe = target_fe.sort_values('r', ascending=False)
    target_fe.reset_index(drop=True, inplace=True)
    all_features = target_fe.loc[target_fe.r >= r, 'feature']
    print('End with features number:', len(all_features))
    return all_features
