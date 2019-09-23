# coding: utf-8
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import KFold,train_test_split
import lightgbm as lgb
import pandas as pd
import numpy as np
from math import pi, modf
import os, gc, time
import random, logging


def feat(data):
    def get_demical(x):
        return modf(x)[0]
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
    features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d', 'Hu-dHv', 'theta_b-u','theta_T',
            'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
            'Amplitude','logf']
    train_x = train_x[features]
    return train_x

features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d', 'Hu-dHv', 'theta_b-u','theta_T',
            'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
            'Amplitude','logf']

# 读取原始数据
def load_raw_data():
    data_dir = '/nfs/cold_project/wangchenxi/mathmodel/train_set/'
    files = os.listdir(data_dir)
    train_data=[]
    for f in files:
        # print(f)
        train_data.append(pd.read_csv(os.path.join(data_dir, f)))
    train_data=pd.concat(train_data).reset_index(drop=True)
    
    return train_data

#训练数据转化
def feat_ext(train_data):
    train_set=[]
    chunk=1000000
    for i in range(1,14):
        train_x=feat(train_data[(i-1)*chunk:i*chunk])
        train_set.append(train_x)
    train_set=pd.concat(train_set)
    train_set['label']=train_data['RSRP']
    return train_set


train_data = load_raw_data()
train_set = feat_ext(train_data)
del train_data
gc.collect()
params = {
    'learning_rate': 0.15,
    'boosting_type': 'gbdt',
    'max_bin': 60,
    'objective': 'regression_l2',
    'metric': 'mse',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'num_leaves': 500,
    'verbose': -1,
    'max_depth': -1,
    'reg_alpha':0.2,
    'reg_lambda':0.4,
    'nthread': 8
}
#降低chunk时要同时增大refit次数
chunk=90000
count=0
#先shuffle train_set
train_set = train_set.sample(frac=1).reset_index(drop=True)
gc.collect()
for i in range(1,130):
    #分14部分，留最后一部分当test_set
    train_i=train_set[(i-1)*chunk:i*chunk]
    train_x,train_y=train_i[features],train_i['label']
    if count==0:
        print(count)
        num_round = 2000
        trn_data = lgb.Dataset(train_x,label=train_y)
        clf = lgb.train(params, trn_data, num_round)
    else:
        decay_rate=1/count+1
        clf=clf.refit(train_x,train_y,decay_rate=decay_rate)
    count+=1    

test_i=train_set[129*chunk:]
test_x,test_y=train_i[features],train_i['label'].values

pred_y = clf.predict(test_x)
mse = mean_squared_error(test_y.reshape(-1), pred_y.reshape(-1))
