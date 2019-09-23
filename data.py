from sklearn.metrics import mean_absolute_error,mean_squared_error

import pandas as pd 
import numpy as np
from math import pi, modf
# from featexp import *
import gc
import os


# def feat(data):
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     train_x=data.loc[:,feat]
#     train_x['d']=5*np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     train_x['Hb']=data['Cell Altitude']+data['Cell Building Height']+data['Altitude']
#     train_x['Hue']=data['Altitude']+data['Building Height']
#     train_x['Hb-Hue']=train_x['Hb']-train_x['Hue']
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hue-dHv']=train_x['Hb-Hue']-train_x['dHv']
#     train_x['Hue-dHv']=train_x['Hue']-train_x['dHv']
#     train_x['theta_uc']=np.arctan2(train_x['Hb-Hue'],train_x['d'])
#     train_x['log_Hb']=np.log10(train_x['Hb']+1)
#     train_x['log_Hue']=np.log10(train_x['Hue']+1)
#     train_x['log_df']=np.log10(train_x['d']*data['Frequency Band']+1)
#     train_x['log_Hb-Hue-dHv']=np.log10(abs(train_x['Hb-Hue-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=data['X']-data['Cell X']
#     train_x['dY']=data['Y']-data['Cell Y']
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'].loc[train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hue-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['Amplitude']=train_x['RS Power']*np.sin(train_x['d']/(299792458/data['Frequency Band']/1e6))
#     train_x['Amplitude1']=train_x['RS Power']*np.sin(train_x['Larc']/(299792458/data['Frequency Band']/1e6))
#     #train_x['thetaME_exp']=train_x['thetaME']*np.exp(train_x['thetaME'])
#     #train_x['theta_XY_A_exp']=train_x['theta_XY_A']*np.exp(train_x['theta_XY_A'])
#     #train_x['theta_uc_exp']=train_x['theta_uc']*np.exp(train_x['theta_uc'])
#     feat_drop=['d','Hb','Hue','Hb-Hue','Hb-Hue-dHv','dHv',
#                'Hue-dHv','Azimuth_rad','theta_XY','dX','dY','Larc']
#     train_x=train_x.drop(feat_drop,axis=1)
#     return train_x

# def feat(data):
#     # 特征（修改）
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     train_x=data.loc[:,feat]
#     # 天线和栅格水平距离
#     train_x['d']=5*np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     # 天线海拔
#     # train_x['Hb']=data['Cell Altitude']+data['Cell Building Height']
#     train_x['Hb']=data['Cell Altitude'] + data['Height'] - data['Altitude']
#     # 栅格
#     # train_x['Hue']=data['Altitude']+data['Building Height']
#     train_x['Hue']=data['Building Height']
#     train_x['Hb-Hue']=train_x['Hb']-train_x['Hue']
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     train_x['dHv']=train_x['Hb-Hue']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hue-dHv']=train_x['Hb-Hue']-train_x['dHv']
#     train_x['Hue-dHv']=train_x['Hue']-train_x['dHv']
#     train_x['theta_uc']=np.arctan2(train_x['Hb-Hue'],train_x['d'])
#     # train_x['log_Hb']=np.log10(abs(train_x['Hb'])+1)
#     train_x['log_Hb']=np.log10(train_x['Hb']+1)
#     train_x['log_Hue']=np.log10(train_x['Hue']+1)
#     train_x['log_df']=np.log10(train_x['d']*data['Frequency Band']+1)
#     train_x['log_Hb-Hue-dHv']=np.log10(abs(train_x['Hb-Hue-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=5 * (data['X']-data['Cell X'])
#     train_x['dY']=5 * (data['Y']-data['Cell Y'])
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'].loc[train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hue-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['Amplitude']=train_x['RS Power']*np.sin(train_x['d']/(299792458/data['Frequency Band']/1e6))
#     train_x['Amplitude1']=train_x['RS Power']*np.sin(train_x['Larc']/(299792458/data['Frequency Band']/1e6))
#     #train_x['thetaME_exp']=train_x['thetaME']*np.exp(train_x['thetaME'])
#     #train_x['theta_XY_A_exp']=train_x['theta_XY_A']*np.exp(train_x['theta_XY_A'])
#     #train_x['theta_uc_exp']=train_x['theta_uc']*np.exp(train_x['theta_uc'])
#     feat_drop=['d','Hb','Hue','Hb-Hue','Hb-Hue-dHv','dHv',
#                'Hue-dHv','Azimuth_rad','theta_XY','dX','dY','Larc']
#     train_x=train_x.drop(feat_drop,axis=1)
#     return train_x

# def feat(data):
#     def get_demical(x):
#             return modf(x)[0]
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     lambdaf=299792458/data['Frequency Band']/1e6
#     train_x=data.loc[:,feat]
#     train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     train_x['Hb']=data['Cell Altitude']+data['Cell Building Height']
#     train_x['Hue']=data['Altitude']+data['Building Height']
#     train_x['Hb-Hue']=train_x['Hb']-train_x['Hue']
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hue-dHv']=train_x['Hb-Hue']-train_x['dHv']
#     train_x['Hue-dHv']=train_x['Hue']-train_x['dHv']
#     train_x['theta_uc']=np.arctan2(train_x['Hb-Hue'],train_x['d'])
#     train_x['log_Hb']=np.log10(train_x['Hb']+1)
#     train_x['log_Hue']=np.log10(train_x['Hue']+1) 
#     train_x['log_Hb-Hue-dHv']=np.log10(abs(train_x['Hb-Hue-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=data['X']-data['Cell X']
#     train_x['dY']=data['Y']-data['Cell Y']
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'].loc[train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hue-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['dAmplitude']=data['RS Power']*np.cos(train_x['thetaME'])*np.cos(2*pi*((train_x['d']/lambdaf).map(get_demical)))
#     train_x['LarcAmplitude']=train_x['dAmplitude']*np.sin(train_x['theta_XY_A']/2)*np.cos(2*pi*((train_x['Larc']/lambdaf).map(get_demical)))
#     train_x['dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['dHv']/lambdaf).map(get_demical)))
#     train_x['Hb-Hue-dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['Hb-Hue-dHv']/lambdaf).map(get_demical)))
#     feat_drop=['d','Hb','Hue','Hb-Hue','Hb-Hue-dHv','dHv',
#             'Hue-dHv','Azimuth_rad','theta_XY','dX','dY','Larc']
#     train_x=train_x.drop(feat_drop,axis=1)
#     return train_x

def feat(data):
    def get_demical(x):
        return modf(x)[0]
    feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
    lambdaf=299792458/data['Frequency Band']/1e6
    train_x=data.loc[:,feat]
    train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
    #train_x['lambda']=lambdaf
    #train_x['logFreq']=np.log10(data['Frequency Band']*1e6)
    train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
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
    train_x['log_Hb']=np.log10(train_x['Hb']+1)
    train_x['log_Hu']=np.log10(train_x['Hu']+1) 
    train_x['log_HuA']=np.log10(train_x['HuA']+1)
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
    features=['Frequency Band', 'RS Power', 'Cell Clutter Index', 'Clutter Index', 'd', 'Hu-dHv', 'theta_b-u','theta_T',
                'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
                'Amplitude', 'cos_thetaME']
    train_x = train_x[features]
    return train_x

def load_raw_data(data_dir):
    #读取数据
    file_list = os.listdir(data_dir)
    data=[]
    for f in file_list:
        data.append(pd.read_csv(os.path.join(data_dir, f)))
    data=pd.concat(data).reset_index(drop=True)
    return data


#按Frequency Band分3类,做成dict
def DataDivison(data_set):
    groups=data_set.groupby('Frequency Band')
    data_sets={}
    for key,group in groups:
        # print(key)
        group=group.drop(['Frequency Band'],axis=1)
        data_sets[key]=group
    return data_sets

def load_data(train_data_dir, test_data_dir):
    train_data = load_raw_data(train_data_dir)
    test_data = load_raw_data(test_data_dir)
    return train_data, test_data

def feat_extract(train_data, test_data):
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

    train_sets=DataDivison(train_set)
    test_sets=DataDivison(test_set)
    return train_sets, test_sets