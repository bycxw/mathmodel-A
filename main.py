# coding: utf-8
from model import MLPModel
from data import load_data, feat_extract
import os, logging
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S %a')


path_train='/nfs/cold_project/wangchenxi/mathmodel/train_set/'
path_test='/nfs/cold_project/wangchenxi/mathmodel/test_set/'

train_data, test_data = load_data(path_train, path_test)
train_sets, test_sets = feat_extract(train_data, test_data)

del train_data

features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'd', 'Hu-dHv', 'theta_b-u','theta_T',
            'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
            'Amplitude', 'cos_thetaME']
# dataSetKey=[2604.8, 2624.6, 2585.0]


args = {'feat_dim': None,
        'batch_size': 64,
        'epochs': 2,
        'lr': 0.001
        }

def train_eval(train, targets):
    features=train.columns
    folds = KFold(n_splits=5, shuffle=True, random_state=1420)
    oof = np.zeros(len(train))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, targets.values)):
        if fold_ > 0:
            break
        print("Fold {}".format(fold_))
        #n=len(trn_idx)
        trn_data, trn_label = train.iloc[trn_idx][features].values, targets.iloc[trn_idx].values
        val_data, val_label = train.iloc[val_idx][features].values, targets.iloc[val_idx].values
        model = MLPModel(args)
        model.train(trn_data, trn_label, val_data, val_label)

        oof[val_idx] = model.evaluate(val_data, val_label)
        
        # fold_importance_df = pd.DataFrame()
        # fold_importance_df["Feature"] = features
        # fold_importance_df["importance"] = clf.feature_importance()
        # fold_importance_df["fold"] = fold_ + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        mae = mean_absolute_error(targets.iloc[val_idx], oof[val_idx])
        mse = mean_squared_error(targets.iloc[val_idx], oof[val_idx])
        
        logging.info('mae: {:<8.5f}'.format(mae))
        logging.info('mse: {:<8.5f}'.format(mse))
    
    # print("CV score: {:<8.5f}".format(Score))

dataSetKey=[2585.0, 2604.8, 2624.6]
for key in dataSetKey:
    args['feat_dim'] = len(features)
    print('feat_dim: {}'.format(len(features)))
    train_eval(train_sets[key][features], train_sets[key]['label'])
    

