import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from utils_multi_step_prediction import Dataset, Level_min, SplitTrainValidTest, Data, Normalization, NormalizeMinMaxTest  
from utils_multi_step_prediction import NormalizeMinMax, ShapeY, Vector, GridSearch, Metrics, Graph, area_under_curve, time_weighting_root_mean_squared_error

   

start = time.time()
path = '/home/.../Data'
folder = "/home/.../"
percentage = 0.5
train_pc, valid_pc, test_pc = 0.7, 0.1, 0.2


data, metadata, names = Dataset(path) 
new_data = Level_min(data)
data_trn, data_val, data_tst, trn_name, val_name, tst_name, metadata_trn, metadata_val, metadata_tst = SplitTrainValidTest(new_data, metadata, names, train_pc, valid_pc)
print('data_trn: ', len(data_trn))
print('data_val: ', len(data_val))
print('data_tst: ', len(data_tst))

Data_type = object    

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

#GridSearch(vec_trn, y_trn)
    
#regressor = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, random_state=0)
regressor = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=1, random_state=0)
#regressor = xgb.XGBRegressor(max_depth=3, n_estimators=10, learning_rate=0.5)
regressor.fit(vec_trn, y_trn)
    
print("-------------Train-RF------------")
Ypred_trn = regressor.predict(vec_trn)
Metrics(y_trn, Ypred_trn)
area_under_curve(y_trn, Ypred_trn, time_y_trn)
time_weighting_root_mean_squared_error(y_trn, Ypred_trn, time_y_trn)

print("-------------VAL-RF------------")
Ypred_val = regressor.predict(vec_val) 
Metrics(y_val, Ypred_val)
area_under_curve(y_val, Ypred_val, time_y_val)
time_weighting_root_mean_squared_error(y_val, Ypred_val, time_y_val)

print("-------------Test-RF------------")
Ypred_tst = regressor.predict(vec_tst) 
Metrics(y_tst, Ypred_tst)
area_under_curve(y_tst, Ypred_tst, time_y_tst)
time_weighting_root_mean_squared_error(y_tst, Ypred_tst, time_y_tst)

end = time.time()
print('Time: ', end-start)

np.save(folder +'y_tst_{}.npy'.format(int(percentage*100)), y_tst)    
np.save(folder +'time_tst_{}.npy'.format(int(percentage*100)), time_tst)  
np.save(folder +'time_y_tst_{}.npy'.format(int(percentage*100)), time_y_tst)  
np.save(folder +'x_data_tst_{}.npy'.format(int(percentage*100)), x_data_tst)  
np.save(folder +'Ypred_tst_{}.npy'.format(int(percentage*100)), Ypred_tst)  
np.save(folder +'tst_name_{}.npy'.format(int(percentage*100)), tst_name)  
    
Graph(y_val, time_val, time_y_val, x_data_val, Ypred_val, val_name, folder + "Val")
Graph(y_tst, time_tst, time_y_tst, x_data_tst, Ypred_tst, tst_name, folder + "Test")
Graph(y_trn, time_trn, time_y_trn, x_data_trn, Ypred_trn, trn_name, folder + "Train")
