import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from utils_multi_step_prediction import Dataset, Level_min, SplitTrainValidTest, Data, Normalization, NormalizeMinMaxTest, NormalizeMinMax, ShapeY, Vector  


#   REFERENCES: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#               https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
#               https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py


def MDI(regressor, feature_names, folder_save, name_save):
    features_importances = pd.Series(regressor.feature_importances_, index=feature_names)  
    fig, ax = plt.subplots(figsize=(14, 4))
    features_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.grid()
    plt.savefig(folder_save + name_save + ".png", dpi=600)


def Permutation(regressor, X_test, y_test, feature_names, folder_save, name_save):
    result = permutation_importance(
        regressor, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )   
    features_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots(figsize=(14, 4))
    features_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.grid()
    plt.savefig(folder_save + name_save + ".png", dpi=600)



start = time.time()
path = '/home/.../Data'
folder = "/home/.../"
percentage = 0.5
train_pc, valid_pc, test_pc = 0.7, 0.1, 0.2

Data_type = object   

data, metadata, names = Dataset(path) 
new_data = Level_min(data)
data_trn, data_val, data_tst, trn_name, val_name, tst_name, metadata_trn, metadata_val, metadata_tst = SplitTrainValidTest(new_data, metadata, names, train_pc, valid_pc)

time_trn, x_trn, y_trn, D_trn, L_trn, R_trn, P_trn, cant_trn, time_y_trn = Data(data_trn, metadata_trn, percentage)         
zmin, zmax = Normalization(x_trn, y_trn)
x_data_trn = NormalizeMinMaxTest(x_trn, zmax, zmin)
y_trn = np.array(ShapeY(NormalizeMinMaxTest(y_trn, zmax, zmin)), dtype=Data_type)
D_trn, dose_trn_max, dose_trn_min = NormalizeMinMax(D_trn)
vec_trn = np.array(Vector(time_trn, x_data_trn, D_trn, L_trn, R_trn, P_trn), dtype=Data_type) 
    
time_val, x_val, y_val, dose_val, L_val, R_val, P_val, cant_val, time_y_val = Data(data_val, metadata_val, percentage)
x_data_val = NormalizeMinMaxTest(x_val, zmax, zmin)
y_val = np.array(ShapeY(NormalizeMinMaxTest(y_val, zmax, zmin)), dtype=Data_type)
dose_val = NormalizeMinMaxTest(dose_val, dose_trn_max, dose_trn_min)
vec_val = np.array(Vector(time_val, x_data_val, dose_val, L_val, R_val, P_val), dtype=Data_type) 
    
time_tst, x_tst, y_tst, D_tst, L_tst, R_tst, P_tst, cant_tst, time_y_tst= Data(data_tst, metadata_tst, percentage)
x_data_tst = NormalizeMinMaxTest(x_tst, zmax, zmin )
y_tst = np.array(ShapeY(NormalizeMinMaxTest(y_tst, zmax, zmin)), dtype=Data_type)
D_tst = NormalizeMinMaxTest(D_tst, dose_trn_max, dose_trn_min)
vec_tst = np.array(Vector(time_tst, x_data_tst, D_tst, L_tst, R_tst, P_tst), dtype=Data_type)

feature_names = ['Dose','L_0','L_1','L_2','L_3','R_0','R_1','R_2','R_3','R_4','R_5','P_0','P_1','P_2','P_3','P_4','P_5','P_6','P_7','P_8','T_0','T_1','T_2','T_3','T_4','T_5','T_6','T_7','T_8','T_9','T_10','T_11','T_12','T_13','T_14','T_15','T_16','T_17','T_18','T_19','T_20','T_21','T_22','T_23','T_24','T_25','T_26','T_27','T_28','T_29','T_30','T_31','S_0','S_1','S_2','S_3','S_4','S_5','S_6','S_7','S_8','S_9','S_10','S_11','S_12','S_13','S_14','S_15','S_16','S_17','S_18','S_19','S_20','S_21','S_22','S_23','S_24','S_25','S_26','S_27','S_28','S_29','S_30','S_31']

folder_save = "/home/.../"

regressor1 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, random_state=0)
regressor1.fit(vec_trn, y_trn)
MDI(regressor1, feature_names, folder_save, 'MDI_DecisionTree')
Permutation(regressor1, vec_tst, y_tst, feature_names, folder_save, 'Permutation_DecisionTree')

regressor2 = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=1, random_state=0)
regressor2.fit(vec_trn, y_trn)
MDI(regressor2, feature_names, folder_save, 'MDI_RandomForest')
Permutation(regressor2, vec_tst, y_tst, feature_names, folder_save, 'Permutation_RandomForest')

regressor3 = xgb.XGBRegressor(max_depth=3, n_estimators=10, learning_rate=0.5)  
regressor3.fit(vec_trn, y_trn)
MDI(regressor3, feature_names, folder_save, 'MDI_XGB')
Permutation(regressor3, vec_tst, y_tst, feature_names, folder_save, 'Permutation_XGB')

