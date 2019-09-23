from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import KFold,train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
import lightgbm as lgb
import pandas as pd 
import numpy as np
from math import pi, modf
# from featexp import *
import gc
import os, time
import random, logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S %a')

# path_train='D:/data_mywork/math_modeling/train_set/'
# path_test='D:/data_mywork/math_modeling/test_set/'
path_train='/nfs/cold_project/wangchenxi/mathmodel/train_set/'
path_test='/nfs/cold_project/wangchenxi/mathmodel/test_set/'
path_dir=os.listdir(path_train)
path_dir1=os.listdir(path_test)

#改进版
def get_demical(x):
    return modf(x)[0]
def feat(data):
    feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
    lambdaf=299792458/data['Frequency Band']/1e6
    train_x=data.loc[:,feat]
    train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
    train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
    train_x['logf']=39.9*np.log10(data['Frequency Band']*1e6)
    train_x['Hb']=data['Cell Altitude']+data['Height']
    train_x['Hu']=data['Altitude']+data['Building Height']
    train_x['HuA']=data['Altitude']                               
    train_x['Hb-Hu']=train_x['Hb']-train_x['Hu']
    train_x['Hb-HuA']=train_x['Hb']-train_x['HuA']                               
    train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
    train_x['Hb-Hu-dHv']=train_x['Hb-Hu']-train_x['dHv']
    train_x['Hb-HuA-dHv']=train_x['Hb-HuA']-train_x['dHv']
    train_x['Hu-dHv']= train_x['HuA']-train_x['dHv']                              
    train_x['HuA-dHv']=train_x['HuA']-train_x['dHv']
    
    train_x['theta_b-u']=np.arctan2(train_x['Hb-Hu'],train_x['d'])-train_x['thetaME']
    train_x['theta_b-uA']=np.arctan2(train_x['Hb-HuA'],train_x['d'])-train_x['thetaME']
    #train_x['reflectSign1']=np.sign(train_x['theta_b-u'])
    #train_x['reflectSign2']=np.sign(train_x['theta_b-uA']) 
    train_x['log_d']=44.9*np.log10(train_x['d']+1)
    train_x['log_Hb']=13.82*np.log10(train_x['Hb']+1)
    train_x['log_Hu']=6.55*np.log10(train_x['Hu']+1)*np.log10(train_x['d']+1) 
    train_x['log_HuA']=6.55*np.log10(train_x['HuA']+1)*np.log10(train_x['d']+1) 
    train_x['log_dHv']=np.log10(abs(train_x['dHv'])+1)
    train_x['log_Hb-Hu']=np.log10(abs(train_x['Hb-Hu'])+1)
    train_x['log_Hb-HuA']=np.log10(abs(train_x['Hb-HuA'])+1) 
    train_x['log_Hb-Hu-dHv']=np.log10(abs(train_x['Hb-Hu-dHv'])+1)
    train_x['log_Hb-HuA-dHv']=np.log10(abs(train_x['Hb-HuA-dHv'])+1)
    
    train_x['Azimuth_rad']=np.radians(data['Azimuth'])
    train_x['dX']=data['X']-data['Cell X']
    train_x['dY']=data['Y']-data['Cell Y']
    train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
    train_x['theta_XY'][train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
    train_x['theta_XY'][(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
    train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
    train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
    train_x['theta_T']=np.arctan2(train_x['Hu-dHv'],train_x['Larc'])
    train_x['theta_T1']=np.arctan2(train_x['HuA-dHv'],train_x['Larc'])
    train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
    train_x['Amplitude']=data['RS Power']*np.cos(train_x['thetaME'])*np.cos(train_x['theta_XY_A'])
    train_x['dAmplitude']=train_x['Amplitude']*np.cos(2*pi*((train_x['d']/lambdaf).map(get_demical)))
    train_x['dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['dHv']/lambdaf).map(get_demical)))
    
    train_x['cos_thetaME']=np.cos(train_x['thetaME'])
    train_x['cos_theta_T']=np.cos(train_x['theta_T'])
    train_x['cos_theta_b-u']=np.cos(train_x['theta_b-u'])
    train_x['cos_theta_b-uA']=np.cos(train_x['theta_b-uA'])
    train_x['cos_theta_XY_A']=np.cos(train_x['theta_XY_A'])
    feat_drop=['Azimuth_rad','theta_XY','dX','dY']
    train_x=train_x.drop(feat_drop,axis=1)
    features=['Frequency Band', 'RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d','log_Hb','log_Hu','log_HuA','Hu-dHv','HuA-dHv',
            'log_dHv', 'Hb-Hu-dHv','Hb-HuA-dHv', 'theta_XY_A', 'logLarc','Amplitude', 'thetaME',
            'cos_theta_T', 'theta_b-uA','theta_b-u']
    train_x = train_x[features]
    return train_x
# #特征版本1
# features1=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'd', 'Hu-dHv', 'theta_b-u','theta_T',
#  'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
#  'Amplitude']
#特征版本2
# features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d','log_Hb','log_Hu','log_HuA','HuA-dHv',
#   'log_dHv', 'log_Hb-Hu-dHv','log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','Amplitude', 'cos_thetaME',
#  'cos_theta_T', 'cos_theta_b-uA','cos_theta_b-u']
# #特征版本3
features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d','log_Hb','log_Hu','log_HuA','Hu-dHv','HuA-dHv',
  'log_dHv', 'Hb-Hu-dHv','Hb-HuA-dHv', 'theta_XY_A', 'logLarc','Amplitude', 'thetaME',
 'cos_theta_T', 'theta_b-uA','theta_b-u']

#读取训练数据
train_data=[]
for f in path_dir:
    train_data.append(pd.read_csv(path_train+f))
train_data=pd.concat(train_data).reset_index(drop=True) 
#读取测试数据
test_data=[]
for f in path_dir1:
    test_data.append(pd.read_csv(path_test+f))
test_data=pd.concat(test_data).reset_index(drop=True) 

CellIndex=test_data['Cell Index']

#训练数据转化
train_set=[]
chunk=1000000
for i in range(1,14):
    train_x=feat(train_data[(i-1)*chunk:i*chunk])
    train_set.append(train_x)
train_set=pd.concat(train_set)
train_set['label']=train_data['RSRP']
#测试数据转化
test_set=feat(test_data)
del train_data

#按Frequency Band分3类,做成dict
def DataDivison(data_set):
    groups=data_set.groupby('Frequency Band')
    data_sets={}
    for key,group in groups:
        # print(key)
        group=group.drop(['Frequency Band'],axis=1)
        data_sets[key]=group
    return data_sets

#分别建模
train_sets=DataDivison(train_set)
test_sets=DataDivison(test_set)


del train_set

#para
# def display_importances(feature_importance_df_):
#     cols = feature_importance_df_[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:40].index
#     best_features = feature_importance_df_.loc[feature_importance_df_.Feature.isin(cols)]
#     plt.figure(figsize=(8, 10))
#     sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
#     plt.title('LightGBM Features (avg over folds)')
#     plt.tight_layout()
#     plt.show()
    


# params = {
#     'learning_rate': 0.08,
#     'boosting_type': 'gbdt',
#     'objective': 'regression_l2',
#     'metric': 'mse',
#     'feature_fraction': 0.6,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'num_leaves': 200,
#     'verbose': -1,
#     'max_depth': -1,
#     'reg_alpha':0.2,
#     'reg_lambda':0.4,
#     'nthread': -1
# }

params={
            'num_round': 10000, 
            'early_stopping_rounds': 1000,
            'num_leaves': 500,
            'max_bin': 240,
            # 'reg_alpha':0.1,
            #'reg_lambda':0.1,
            'lambda_l1':0.5,
            'lambda_l2':0.5,
            'min_data_in_leaf': 50,
            'learning_rate': 0.1,
            'min_sum_hessian_in_leaf': 0.000446,
            'bagging_fraction': 0.61, 
            'bagging_freq': 10, 
            'max_depth': -1,
            'seed': 6666,
            'feature_fraction_seed': 6666,
            'feature_fraction': 0.61,
            'bagging_seed': 6666,
            'boosting_type': 'gbdt',
            'verbose': 1,
            'metric':'mse',
            'objective':'regression', 
            'n_jobs':-1
    }

def lgb_reg(key, params,train,targets,test=None, is_train=True, model_dir=None, sample_ratio=None):
    logging.info(params)
    features=train.columns
    folds = KFold(n_splits=5, shuffle=True, random_state=1420)
    oof = np.zeros(len(train))
    predictions = None
    # predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()
    if sample_ratio != None:
        idx = random.sample(range(len(train)), int(sample_ratio*len(train)))
        train = train.iloc[idx]
        targets = targets.iloc[idx]
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, targets.values)):
        if fold_ > 0:
            break
        
        logging.info("Fold {}".format(fold_))
        #n=len(trn_idx)
        
        trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=targets.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx][features], label=targets.iloc[val_idx])

        if is_train:
            
            clf = lgb.train(params, trn_data, valid_sets = [trn_data, val_data], verbose_eval=100, categorical_feature=['Cell Clutter Index', 'Clutter Index'])
            if model_dir != None:
                model_path = os.path.join(model_dir, 'lgb_model_{}'.format(key))
                clf.save_model(model_path.format(key), num_iteration=clf.best_iteration)
            # else:
            #     model_path = os.path.join('./modelfile/', 'lgb_model_{}'.format(key))
            
        else:
            model_path = os.path.join(model_dir, 'lgb_model_{}'.format(key))
            clf = lgb.Booster(model_file=model_path)
        oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        # predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
        mse = mean_squared_error(targets.iloc[val_idx], oof[val_idx])
        mae = mean_absolute_error(targets.iloc[val_idx], oof[val_idx])
        
        logging.info('mse: {:<8.5f}'.format(mse))
        logging.info('mae: {:<8.5f}'.format(mae))
    # Score=mean_absolute_error(targets, oof)
    # print("CV score: {:<8.5f}".format(Score))

    return oof,predictions,feature_importance_df



# features=train_sets[2604.8].columns.tolist()[:-1]


#oof,predictions,feature_importance_df=lgb_reg(params,trainSet1[features][:100000],trainSet1['target'][:100000],testSet1[features][100000:120000])

#按类别训练并预测
result=pd.DataFrame()
result['CellIndex']=CellIndex
result['predict'] = np.nan

dataSetKey=[2604.8, 2624.6, 2585.0]
# print(test_sets.keys())
# for key in test_sets.keys():

is_train = True
if is_train:
    model_dir = os.path.join('./modelfile/', str(int(time.time())))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info('Make dir {} successfully!'.format(model_dir))
    pass
else:
    modle_num = '156xxxxx'
    model_dir = os.path.join('./modelfile/', model_num)
for key in dataSetKey:
# for key in [2585.0]:
    logging.info(key)
    if key == 2585.0:
        ratio = None
    else:
        ratio = None
    oof,predictions,feature_importance_df=lgb_reg(key, params,train_sets[key][features],train_sets[key]['label'], is_train=is_train, model_dir=model_dir, sample_ratio=ratio)
    # oof,predictions,feature_importance_df=lgb_reg(params,train_sets[key][features],train_sets[key]['label'],test_sets[key][features])
    # result['predict'].loc[test_sets[key].index]=predictions
# result.to_csv(path_result+'result.csv')
# result.to_csv('result.csv')

