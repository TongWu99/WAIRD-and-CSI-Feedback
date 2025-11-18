'''
Copyright (C) 2021. Huawei Technologies Co., Ltd.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
'''

import warnings
import numpy as np
import multiprocessing
from PIL import Image
warnings.filterwarnings('ignore')

numCores = multiprocessing.cpu_count() # number of cores to do generation job, decrease this number if your computer has less memory or available cores

# parameters for generating dataset
carrierFreq = '2_6' # for example, 2_6 for 2.6G, 60_0 for 60.0G
BWGHz = 0.00936 # bandwidth in GHz
subcarriers = 624 # number of subcarriers
carrierSampleInterval = 12 # sample subcarriers with this interval to save computation time 
sampledCarriers = int(subcarriers/carrierSampleInterval) # number of sampled subcarriers for deep learning 
Nt = [1, 8, 4] # BS antenna array in [x,y,z] axis, e.g., [1, 8, 8], [1, 32, 4]
Nr = [1, 1, 1] # UE antenna array in [x,y,z] axis, e.g., [2, 2, 1], [4, 2, 1]
spacing_t = [0.5, 0.5, 2] # transmitter antenna spacing in wavelength
spacing_r = [0.5, 0.5, 2] # receiver antenna spacing in wavelength
Pattern_t = {'Power':0} # omni antenna type for default, transmitter power 0 dBm
Basis_t = np.eye(3) # antenna basis rotation, no rotation for default
Basis_r = np.eye(3) # antenna basis rotation, no rotation for default
saveAsArray = True # save channel as numpy array if True
saveAsImage = True # save channel as image if True
saveCombinedArray = True # save all users' CSI in one combined npy file per environment if True
maxPathNum = 1000 # should be >0, max Path number for every BS-UE link, a large number such as 1000 means no limits
scenario = 2 # select a scenario to generate channel, the detailed description of scenarios are listed below
scenarioFolder = f'data/scenario_{scenario}/' # folder of scenario
generatedFolder = f'data/RawData/generated_{scenario}_{carrierFreq}_{maxPathNum}_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz*1000)}_{sampledCarriers}/'

# 自定义环境选择 (可选)
# 如果这个列表不为空, generator 将会使用这些指定的环境ID.
# 示例: customEnvList = ['00247', '00812']
customEnvList = ['00873']

if scenario==1:
    # scenario_1: sparse UE drop in lots of environments
    # max 10000 envs, 5 BS and 30 UE drops can be selected for every environment
    ENVnum = 1000 # number of environments to pick, max is 10000
    BSlist = list(range(5)) # BS index range 0~4 per environment, e.g., [0] picks BS_0, [2,4] picks BS_2 and BS_4
    UElist = list(range(30)) # UE index range 0~29 per environment, e.g., [0] picks UE_0, [2,17,26] picks UE_2, UE_17 and UE_26
    BSnum = len(BSlist) # number of BS per environment, max is 5
    UEnum = len(UElist) # number of UE per environment, max is 30
elif scenario==2:
    # scenario_2: dense UE drop in some environments
    # max 100 envs, 1 BS and 10000 UE drops can be selected for every environment
    ENVnum = 1 # number of environments to pick, max is 100
    BSlist = list(range(1)) # BS index range 0~0 per environment, e.g., [0] picks BS_0
    UElist = list(range(10000)) # UE index range 0~9999 per environment, e.g., [0] picks UE_0, [2,170,2600] picks UE_2, UE_170 and UE_2600
    BSnum = len(BSlist) # number of BS per environment, max is 1
    UEnum = len(UElist) # number of UE per environment, max is 10000
else:
    raise('More scenarios are in preparation.')