import numpy as np
import pandas as pd
from math import pi, modf

# def feat(data):
#     def get_demical(x):
#             return modf(x)[0]
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     lambdaf=299792458/data['Frequency Band']/1e6
#     train_x=data.loc[:,feat]
#     train_x['d']=5*np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
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

# def feat(data):
#     def get_demical(x):
#         return modf(x)[0]
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     lambdaf=299792458/data['Frequency Band']/1e6
#     train_x=data.loc[:,feat]
#     train_x['lambda']=lambdaf
#     train_x['logFreq']=np.log10(data['Frequency Band']*1e6)
#     train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     train_x['Hb']=data['Cell Altitude']+data['Height']
#     train_x['Hu']=data['Altitude']+data['Building Height']
#     train_x['HuA']=data['Altitude']                               
#     train_x['Hb-Hu']=train_x['Hb']-train_x['Hu']
#     train_x['Hb-HuA']=train_x['Hb']-train_x['HuA']                               
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hu-dHv']=train_x['Hb-Hu']-train_x['dHv']
#     train_x['Hb-HuA-dHv']=train_x['Hb-HuA']-train_x['dHv']
#     train_x['Hu-dHv']= train_x['HuA']-train_x['dHv']                              
#     train_x['HuA-dHv']=train_x['HuA']-train_x['dHv']
#     train_x['theta_b-u']=np.arctan2(train_x['Hb-Hu'],train_x['d'])-train_x['thetaME']
#     train_x['theta_b-uA']=np.arctan2(train_x['Hb-HuA'],train_x['d'])-train_x['thetaME']
#     train_x['reflectSign1']=np.sign(train_x['theta_b-u'])
#     train_x['reflectSign2']=np.sign(train_x['theta_b-uA'])                               
#     #train_x['log_Hb']=np.log10(train_x['Hb']+1)
#     #train_x['log_Hu']=np.log10(train_x['Hu']+1) 
#     #train_x['log_Hb-Hu-dHv']=np.log10(abs(train_x['Hb-Hu-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=data['X']-data['Cell X']
#     train_x['dY']=data['Y']-data['Cell Y']
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'][train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'][(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hu-dHv'],train_x['Larc'])
#     train_x['theta_T1']=np.arctan2(train_x['HuA-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['dAmplitude']=data['RS Power']*np.cos(train_x['thetaME'])*np.cos(2*pi*((train_x['d']/lambdaf).map(get_demical)))
#     train_x['LarcAmplitude']=train_x['dAmplitude']*np.sin(train_x['theta_XY_A']/2)*np.cos(2*pi*((train_x['Larc']/lambdaf).map(get_demical)))
#     #train_x['dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['dHv']/lambdaf).map(get_demical)))
#     #train_x['Hb-Hu-dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['Hb-Hu-dHv']/lambdaf).map(get_demical)))
#     #train_x['cos_thetaME']=np.cos(train_x['thetaME'])
#     #train_x['cos_theta_T']=np.cos(train_x['theta_T'])
#     train_x['cos_theta_b-u']=np.cos(train_x['theta_b-u'])
#     train_x['cos_theta_XY_A']=np.cos(train_x['theta_XY_A'])
#     feat_drop=['Azimuth_rad','theta_XY','dX','dY']
#     train_x=train_x.drop(feat_drop,axis=1)
#     return train_x

# def feat(data):
#     def get_demical(x):
#         return modf(x)[0]
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     lambdaf=299792458/data['Frequency Band']/1e6
#     train_x=data.loc[:,feat]
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     #train_x['lambda']=lambdaf
#     #train_x['logFreq']=np.log10(data['Frequency Band']*1e6)
#     train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     train_x['Hb']=data['Cell Altitude']+data['Height']
#     train_x['Hu']=data['Altitude']+data['Building Height']
#     train_x['HuA']=data['Altitude']                               
#     train_x['Hb-Hu']=train_x['Hb']-train_x['Hu']
#     train_x['Hb-HuA']=train_x['Hb']-train_x['HuA']                               
#     train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hu-dHv']=train_x['Hb-Hu']-train_x['dHv']
#     train_x['Hb-HuA-dHv']=train_x['Hb-HuA']-train_x['dHv']
#     train_x['Hu-dHv']= train_x['HuA']-train_x['dHv']                              
#     train_x['HuA-dHv']=train_x['HuA']-train_x['dHv']
    
#     train_x['theta_b-u']=np.arctan2(train_x['Hb-Hu'],train_x['d'])-train_x['thetaME']
#     train_x['theta_b-uA']=np.arctan2(train_x['Hb-HuA'],train_x['d'])-train_x['thetaME']
#     #train_x['reflectSign1']=np.sign(train_x['theta_b-u'])
#     #train_x['reflectSign2']=np.sign(train_x['theta_b-uA'])  
#     train_x['log_Hb']=np.log10(train_x['Hb']+1)
#     train_x['log_Hu']=np.log10(train_x['Hu']+1) 
#     train_x['log_HuA']=np.log10(train_x['HuA']+1)
#     train_x['log_dHv']=np.log10(abs(train_x['dHv'])+1)
#     train_x['log_Hb-Hu']=np.log10(abs(train_x['Hb-Hu'])+1)
#     train_x['log_Hb-HuA']=np.log10(abs(train_x['Hb-HuA'])+1) 
#     train_x['log_Hb-Hu-dHv']=np.log10(abs(train_x['Hb-Hu-dHv'])+1)
#     train_x['log_Hb-HuA-dHv']=np.log10(abs(train_x['Hb-HuA-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=data['X']-data['Cell X']
#     train_x['dY']=data['Y']-data['Cell Y']
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'][train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'][(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hu-dHv'],train_x['Larc'])
#     train_x['theta_T1']=np.arctan2(train_x['HuA-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['Amplitude']=data['RS Power']*np.cos(train_x['thetaME'])*np.cos(train_x['theta_XY_A'])
#     train_x['dAmplitude']=train_x['Amplitude']*np.cos(2*pi*((train_x['d']/lambdaf).map(get_demical)))
#     train_x['dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['dHv']/lambdaf).map(get_demical)))
    
#     train_x['cos_thetaME']=np.cos(train_x['thetaME'])
#     train_x['cos_theta_T']=np.cos(train_x['theta_T'])
#     train_x['cos_theta_b-u']=np.cos(train_x['theta_b-u'])
#     train_x['cos_theta_b-uA']=np.cos(train_x['theta_b-uA'])
#     train_x['cos_theta_XY_A']=np.cos(train_x['theta_XY_A'])
#     feat_drop=['Azimuth_rad','theta_XY','dX','dY']
#     train_x=train_x.drop(feat_drop,axis=1)
#     features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'd', 'Hu-dHv', 'theta_b-u','theta_T',
#                 'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
#                 'Amplitude', 'cos_thetaME']
#     train_x = train_x[features]
#     return train_x

# #改进版
# def get_demical(x):
#     return modf(x)[0]
# def feat(data):
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     lambdaf=299792458/data['Frequency Band']/1e6
#     train_x=data.loc[:,feat]
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     train_x['logf']=39.9*np.log10(data['Frequency Band']*1e6)
#     train_x['Hb']=data['Cell Altitude']+data['Height']
#     train_x['Hu']=data['Altitude']+data['Building Height']
#     train_x['HuA']=data['Altitude']                               
#     train_x['Hb-Hu']=train_x['Hb']-train_x['Hu']
#     train_x['Hb-HuA']=train_x['Hb']-train_x['HuA']                               
#     train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hu-dHv']=train_x['Hb-Hu']-train_x['dHv']
#     train_x['Hb-HuA-dHv']=train_x['Hb-HuA']-train_x['dHv']
#     train_x['Hu-dHv']= train_x['HuA']-train_x['dHv']                              
#     train_x['HuA-dHv']=train_x['HuA']-train_x['dHv']
    
#     train_x['theta_b-u']=np.arctan2(train_x['Hb-Hu'],train_x['d'])-train_x['thetaME']
#     train_x['theta_b-uA']=np.arctan2(train_x['Hb-HuA'],train_x['d'])-train_x['thetaME']
#     #train_x['reflectSign1']=np.sign(train_x['theta_b-u'])
#     #train_x['reflectSign2']=np.sign(train_x['theta_b-uA']) 
#     train_x['log_d']=44.9*np.log10(train_x['d']+1)
#     train_x['log_Hb']=13.82*np.log10(train_x['Hb']+1)
#     train_x['log_Hu']=6.55*np.log10(train_x['Hu']+1)*np.log10(train_x['d']+1) 
#     train_x['log_HuA']=6.55*np.log10(train_x['HuA']+1)*np.log10(train_x['d']+1) 
#     train_x['log_dHv']=np.log10(abs(train_x['dHv'])+1)
#     train_x['log_Hb-Hu']=np.log10(abs(train_x['Hb-Hu'])+1)
#     train_x['log_Hb-HuA']=np.log10(abs(train_x['Hb-HuA'])+1) 
#     train_x['log_Hb-Hu-dHv']=np.log10(abs(train_x['Hb-Hu-dHv'])+1)
#     train_x['log_Hb-HuA-dHv']=np.log10(abs(train_x['Hb-HuA-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=data['X']-data['Cell X']
#     train_x['dY']=data['Y']-data['Cell Y']
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'][train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'][(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hu-dHv'],train_x['Larc'])
#     train_x['theta_T1']=np.arctan2(train_x['HuA-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['Amplitude']=data['RS Power']*np.cos(train_x['thetaME'])*np.cos(train_x['theta_XY_A'])
#     train_x['dAmplitude']=train_x['Amplitude']*np.cos(2*pi*((train_x['d']/lambdaf).map(get_demical)))
#     train_x['dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['dHv']/lambdaf).map(get_demical)))
    
#     train_x['cos_thetaME']=np.cos(train_x['thetaME'])
#     train_x['cos_theta_T']=np.cos(train_x['theta_T'])
#     train_x['cos_theta_b-u']=np.cos(train_x['theta_b-u'])
#     train_x['cos_theta_b-uA']=np.cos(train_x['theta_b-uA'])
#     train_x['cos_theta_XY_A']=np.cos(train_x['theta_XY_A'])
#     feat_drop=['Azimuth_rad','theta_XY','dX','dY']
#     train_x=train_x.drop(feat_drop,axis=1)
#     features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d','log_Hb','log_Hu','log_HuA','HuA-dHv',
#             'log_dHv', 'log_Hb-Hu-dHv','log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','Amplitude', 'cos_thetaME',
#             'cos_theta_T', 'cos_theta_b-uA','cos_theta_b-u']
#     train_x = train_x[features]
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
    
    train_x['Larc']=2 * np.sin(train_x['theta_XY_A']/2)*train_x['d']
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
    features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'd', 'Hu-dHv', 'theta_b-u','theta_T',
                'theta_b-uA', 'log_dHv', 'log_Hb-Hu-dHv', 'log_Hb-HuA-dHv', 'theta_XY_A', 'logLarc','log_Hb','log_Hu',
                'Amplitude', 'cos_thetaME']
    train_x = train_x[features]
    return train_x


# def feat(data):
#     def get_demical(x):
#         return modf(x)[0]
#     feat=['Frequency Band','RS Power','Cell Clutter Index','Clutter Index']
#     lambdaf=299792458/data['Frequency Band']/1e6
#     train_x=data.loc[:,feat]
#     train_x['thetaME']=np.radians(data['Mechanical Downtilt']+data['Electrical Downtilt'])
#     train_x['d']=np.sqrt((data['Cell X']-data['X'])**2+(data['Cell Y']-data['Y'])**2)
#     train_x['logf']=39.9*np.log10(data['Frequency Band']*1e6)
#     train_x['Hb']=data['Cell Altitude']+data['Height']
#     train_x['Hu']=data['Altitude']+data['Building Height']
#     train_x['HuA']=data['Altitude']                               
#     train_x['Hb-Hu']=train_x['Hb']-train_x['Hu']
#     train_x['Hb-HuA']=train_x['Hb']-train_x['HuA']                               
#     train_x['dHv']=train_x['Hb']-train_x['d']*np.tan(train_x['thetaME'])
#     train_x['Hb-Hu-dHv']=train_x['Hb-Hu']-train_x['dHv']
#     train_x['Hb-HuA-dHv']=train_x['Hb-HuA']-train_x['dHv']
#     train_x['Hu-dHv']= train_x['HuA']-train_x['dHv']                              
#     train_x['HuA-dHv']=train_x['HuA']-train_x['dHv']
    
#     train_x['theta_b-u']=np.arctan2(train_x['Hb-Hu'],train_x['d'])-train_x['thetaME']
#     train_x['theta_b-uA']=np.arctan2(train_x['Hb-HuA'],train_x['d'])-train_x['thetaME']
#     #train_x['reflectSign1']=np.sign(train_x['theta_b-u'])
#     #train_x['reflectSign2']=np.sign(train_x['theta_b-uA']) 
#     train_x['log_d']=44.9*np.log10(train_x['d']+1)
#     train_x['log_Hb']=13.82*np.log10(train_x['Hb']+1)
#     train_x['log_Hu']=6.55*np.log10(train_x['Hu']+1)*np.log10(train_x['d']+1) 
#     train_x['log_HuA']=6.55*np.log10(train_x['HuA']+1)*np.log10(train_x['d']+1) 
#     train_x['log_dHv']=np.log10(abs(train_x['dHv'])+1)
#     train_x['log_Hb-Hu']=np.log10(abs(train_x['Hb-Hu'])+1)
#     train_x['log_Hb-HuA']=np.log10(abs(train_x['Hb-HuA'])+1) 
#     train_x['log_Hb-Hu-dHv']=np.log10(abs(train_x['Hb-Hu-dHv'])+1)
#     train_x['log_Hb-HuA-dHv']=np.log10(abs(train_x['Hb-HuA-dHv'])+1)
    
#     train_x['Azimuth_rad']=np.radians(data['Azimuth'])
#     train_x['dX']=data['X']-data['Cell X']
#     train_x['dY']=data['Y']-data['Cell Y']
#     train_x['theta_XY']=np.arctan2(train_x['dX'],train_x['dY'])
#     train_x['theta_XY'][train_x['dY']<0]=train_x['theta_XY'].loc[train_x['dY']<0]+pi
#     train_x['theta_XY'][(train_x['dY']>=0)&(train_x['dX']<0)]=train_x['theta_XY'].loc[(train_x['dY']>=0)&(train_x['dX']<0)]+2*pi
#     train_x['theta_XY_A']=train_x['theta_XY']-train_x['Azimuth_rad']
    
#     train_x['Larc']=np.sin(train_x['theta_XY_A']/2)*train_x['d']
#     train_x['theta_T']=np.arctan2(train_x['Hu-dHv'],train_x['Larc'])
#     train_x['theta_T1']=np.arctan2(train_x['HuA-dHv'],train_x['Larc'])
#     train_x['logLarc']=np.log10(abs(train_x['Larc'])+1)
#     train_x['Amplitude']=data['RS Power']*np.cos(train_x['thetaME'])*np.cos(train_x['theta_XY_A'])
#     train_x['dAmplitude']=train_x['Amplitude']*np.cos(2*pi*((train_x['d']/lambdaf).map(get_demical)))
#     train_x['dHvAmplitude']=train_x['dAmplitude']*np.cos(2*pi*((train_x['dHv']/lambdaf).map(get_demical)))
    
#     train_x['cos_thetaME']=np.cos(train_x['thetaME'])
#     train_x['cos_theta_T']=np.cos(train_x['theta_T'])
#     train_x['cos_theta_b-u']=np.cos(train_x['theta_b-u'])
#     train_x['cos_theta_b-uA']=np.cos(train_x['theta_b-uA'])
#     train_x['cos_theta_XY_A']=np.cos(train_x['theta_XY_A'])
#     feat_drop=['Azimuth_rad','theta_XY','dX','dY']
#     train_x=train_x.drop(feat_drop,axis=1)
#     features=['RS Power', 'Cell Clutter Index', 'Clutter Index', 'log_d','log_Hb','log_Hu','log_HuA','Hu-dHv','HuA-dHv',
#             'log_dHv', 'Hb-Hu-dHv','Hb-HuA-dHv', 'theta_XY_A', 'logLarc','Amplitude', 'thetaME',
#             'cos_theta_T', 'theta_b-uA','theta_b-u']
#     train_x = train_x[features]
#     return train_x