import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
import lightgbm as lgb
from math import pi, modf
from data_pre import feat
import os

class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                test_data = pd.read_csv(file_content)
                test_set = feat(test_data)
                if test_data['Frequency Band'][0] == 2585.0:
                    lgb_model_path = os.path.join(self.model_path, 'lgb_model_2585.0')
                    clf = lgb.Booster(model_file=lgb_model_path)
                    pb_data = clf.predict(test_set, num_iteration=clf.best_iteration)
                elif test_data['Frequency Band'][0] == 2604.8:
                    lgb_model_path = os.path.join(self.model_path, 'lgb_model_2604.8')
                    clf = lgb.Booster(model_file=lgb_model_path)
                    pb_data = clf.predict(test_set, num_iteration=clf.best_iteration)
                elif test_data['Frequency Band'][0] == 2624.6:
                    lgb_model_path = os.path.join(self.model_path, 'lgb_model_2624.6')
                    clf = lgb.Booster(model_file=lgb_model_path)
                    pb_data = clf.predict(test_set, num_iteration=clf.best_iteration)
                else:
                    # 取平均
                    lgb_model_path = os.path.join(self.model_path, 'lgb_model_2585.0')
                    clf1 = lgb.Booster(model_file=lgb_model_path)
                    lgb_model_path = os.path.join(self.model_path, 'lgb_model_2604.8')
                    clf2 = lgb.Booster(model_file=lgb_model_path)
                    lgb_model_path = os.path.join(self.model_path, 'lgb_model_2624.6')
                    clf3 = lgb.Booster(model_file=lgb_model_path)
                    
                    pb_data1 = clf1.predict(test_set, num_iteration=clf1.best_iteration).reshape(-1)
                    pb_data2 = clf2.predict(test_set, num_iteration=clf2.best_iteration).reshape(-1)
                    pb_data3 = clf3.predict(test_set, num_iteration=clf3.best_iteration).reshape(-1)
                    pb_data = (pb_data1 + pb_data2 + pb_data3) / 3

                #     print('other frequency! use model 2585.0')
                #     lgb_model_path = os.path.join(self.model_path, 'lgb_model_2585.0')
                # clf = lgb.Booster(model_file=lgb_model_path)

                # if test_data['Frequency Band'][0] == 2604.8:
                #     lgb_model_path = os.path.join(self.model_path, 'lgb_model_2604.8')
                # else:
                #     lgb_model_path = os.path.join(self.model_path, 'lgb_model_2624.6')
                # clf = lgb.Booster(model_file=lgb_model_path)

                

                
                # test_set = feat(test_data)
                # pb_data = clf.predict(test_set, num_iteration=clf.best_iteration)
                input_data = np.array(pb_data.reshape(-1, 1))
                print(file_name, input_data.shape)
                filesDatas.append(input_data)

        filesDatas = np.array(filesDatas,dtype=np.float32).reshape(-1, 1)
        preprocessed_data['inputs'] = filesDatas        
        print("preprocessed_data[\'inputs\'].shape = ", preprocessed_data['inputs'].shape)

        return preprocessed_data


    def _postprocess(self, data):        
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output

        