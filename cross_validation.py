import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import auc
from scipy import integrate
from utils_multi_step_prediction import Dataset, Level_min, Data, Normalization, NormalizeMinMaxTest, NormalizeMinMax, ShapeY, Vector 



def area_under_curve(y_tst, Ypred_tst5, time_y_tst):
    auc_pred_total = 0
    auc_truth_total = 0

    auc_pred_total_trapz = 0
    auc_truth_total_trapz = 0

    auc_pred_total_simpson = 0
    auc_truth_total_simpson = 0

    for i in range(len(time_y_tst)):
        # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        auc_pred = round(auc(time_y_tst[i], Ypred_tst5[i]))
        auc_truth = round(auc(time_y_tst[i], y_tst[i]))
        diff_auc = np.abs(auc_truth - auc_pred)
        
        # REFERENCE: https://numpy.org/devdocs/reference/generated/numpy.trapezoid.html
        auc_pred_trapz = round(np.trapezoid(Ypred_tst5[i], time_y_tst[i], dx=1))
        auc_truth_trapz = round(np.trapezoid(y_tst[i], time_y_tst[i], dx=1))
        diff_trapz = round(np.abs(auc_truth_trapz - auc_pred_trapz))

        # REFERENCIA: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html
        auc_pred_simpson = round(integrate.simpson(Ypred_tst5[i], time_y_tst[i]))
        auc_truth_simpson = round(integrate.simpson(y_tst[i], time_y_tst[i]))
        diff_simpson = round(np.abs(auc_truth_simpson - auc_pred_simpson))

        auc_pred_total = auc_pred_total + auc_pred
        auc_truth_total = auc_truth_total + auc_truth

        auc_pred_total_trapz = auc_pred_total_trapz + auc_pred_trapz
        auc_truth_total_trapz = auc_truth_total_trapz + auc_truth_trapz

        auc_pred_total_simpson = auc_pred_total_simpson + auc_pred_simpson
        auc_truth_total_simpson = auc_truth_total_simpson + auc_truth_simpson

        if (auc_pred == auc_pred_trapz) and (auc_pred == auc_pred_simpson) and (auc_truth == auc_truth_trapz) and (auc_truth == auc_truth_simpson):
            #print('AUC Pred: ', auc_pred, ' Truth: ', auc_truth, ' Difference: ', diff_auc, ' -- AUC trapezoidal Pred: ', auc_pred_trapz, ' Truth: ', auc_truth_trapz, ' Difference: ', diff_trapz, ' -- AUC simpson Pred: ', auc_pred_simpson, ' Truth: ', auc_truth_simpson, ' Difference: ', diff_simpson)
            pass
        else:  
            #print('Attention')
            #print('Attention -- AUC Pred: ', auc_pred, ' Truth: ', auc_truth, ' Difference: ', diff_auc, ' -- AUC trapezoidal Pred: ', auc_pred_trapz, ' Truth: ', auc_truth_trapz, ' Difference: ', diff_trapz, ' -- AUC simpson Pred: ', auc_pred_simpson, ' Truth: ', auc_truth_simpson, ' Difference: ', diff_simpson)
            pass

    mean_auc_pred_total = round(auc_pred_total/len(time_y_tst), 2)
    mean_auc_truth_total = round(auc_truth_total/len(time_y_tst), 2)
    diff_total = round(np.abs(mean_auc_truth_total - mean_auc_pred_total), 2)

    mean_auc_pred_total_trapz = round(auc_pred_total_trapz/len(time_y_tst), 2)
    mean_auc_truth_total_trapz = round(auc_truth_total_trapz/len(time_y_tst), 2)
    diff_total_trapz = round(np.abs(mean_auc_truth_total_trapz - mean_auc_pred_total_trapz), 2)

    mean_auc_pred_total_simpson = round(auc_pred_total_simpson/len(time_y_tst), 2)
    mean_auc_truth_total_simpson = round(auc_truth_total_simpson/len(time_y_tst), 2)
    diff_total_simpson = round(np.abs(mean_auc_truth_total_simpson - mean_auc_pred_total_simpson), 2)

    print('Average auc pred: ', mean_auc_pred_total, ' - trapezoidal: ', mean_auc_pred_total_trapz , ' - simpson: ', mean_auc_pred_total_simpson)
    print('Average auc truth: ', mean_auc_truth_total, ' - trapezoidal: ', mean_auc_truth_total_trapz , ' - simpson: ', mean_auc_truth_total_simpson)
    print('Average difference: ', diff_total, ' - trapezoidal: ', diff_total_trapz , ' - simpson: ', diff_total_simpson)


def MAE_2(gt, pred):
    lista = list()
    for i in range(len(gt)):
        lista.append(np.mean(np.abs([e1 - e2 for e1, e2 in zip(gt[i],pred[i])])))
    print('Mean Absolute Error (MAE):', round(sum(lista)/len(lista),4))
    return round(sum(lista)/len(lista),4)


def EveryThree(name):
    name_final = list()
    for i in range(0,len(name),3):
        name_final.append(name[i])
    return name_final


def Dataa(data, metadata):
    listaData = list()
    listaa = list()
    for i in range(0, len(data)-2, 3):
        listaa.append(i)
        Bret = SumLists(data[i][1], data[i+1][1], data[i+2][1])
        Time = SumLists(data[i][0], data[i+1][0], data[i+2][0])
        listaData.append([np.array(Time), np.array(Bret)])
    return listaData, metadata.iloc[listaa]


def SumLists(lista1, lista2, lista3):
    lista4 = list()
    for i in range(len(lista1)):
        suma = (lista1[i] + lista2[i] + lista3[i]) / 3
        lista4.append(round(suma,3))
    return lista4


def SplitTrainValidTest(data, metadata, name_lista, trn_pc, val_pc):
    new_data = int(len(data) / 3)
    to_val = round(int(new_data * (trn_pc + val_pc)) * 3)

    trn_data = data[:to_val]
    tst_data = data[to_val:]

    trn_name = EveryThree(name_lista[:to_val])
    tst_name = EveryThree(name_lista[to_val:])

    trn_metadata = metadata[:to_val]
    tst_metadata = metadata[to_val:]
    
    listData_trn, metaData_trn = Dataa(trn_data, trn_metadata)
    listData_tst, metaData_tst = Dataa(tst_data, tst_metadata)
    
    return listData_trn, listData_tst, trn_name, tst_name, metaData_trn, metaData_tst


def metrics_mae_auc_rmse(regressor, vec_trn, y_trn, vec_tst, y_tst, time_trn, time_tst):
    regressor.fit(vec_trn, y_trn)

    print("--------------------------", np.shape(vec_trn), " --- ", np.shape(vec_tst), "--------------------------")

    print("-------------Train-RF------------")
    Ypred_trn = regressor.predict(vec_trn)
    mae_trn = MAE_2(y_trn, Ypred_trn)

    area_under_curve(y_trn, Ypred_trn, time_trn)

    time_weighting_root_mean_squared_error(y_trn, Ypred_trn, time_trn)

    print("-------------Test-RF------------")
    Ypred_tst = regressor.predict(vec_tst) 
    mae_tst = MAE_2(y_tst, Ypred_tst)

    area_under_curve(y_tst, Ypred_tst, time_tst)
    
    time_weighting_root_mean_squared_error(y_tst, Ypred_tst, time_tst)

    return Ypred_trn, Ypred_tst, regressor, mae_trn, mae_tst


def concatenation(x_trn_1, x_trn_2, x_trn_3, x_trn_4, x_tst):
    list_1 = np.concatenate([x_trn_2, x_trn_3, x_trn_4, x_tst], axis=0)
    list_2 = np.concatenate([x_trn_1, x_trn_3, x_trn_4, x_tst], axis=0)
    list_3 = np.concatenate([x_trn_1, x_trn_2, x_trn_4, x_tst], axis=0)
    list_4 = np.concatenate([x_trn_1, x_trn_2, x_trn_3, x_tst], axis=0)
    list_5 = np.concatenate([x_trn_1, x_trn_2, x_trn_3, x_trn_4], axis=0)

    return list_1, list_2, list_3, list_4, list_5


def divide(lista):
    div = len(x_trn) // 4
    lista_1 = lista[:div*1]
    lista_2 = lista[div*1:div*2]
    lista_3 = lista[div*2:div*3]
    lista_4 = lista[div*3:div*4+1]

    return lista_1, lista_2, lista_3, lista_4


def normal(x_trn, y_trn, time_trn, D_trn, L_trn, R_trn, P_trn, x_tst, y_tst, time_tst, D_tst, L_tst, R_tst, P_tst):
    zmin, zmax = Normalization(x_trn, y_trn)
    x_data_trn = NormalizeMinMaxTest(x_trn, zmax, zmin)
    y_trn = np.array(ShapeY(NormalizeMinMaxTest(y_trn, zmax, zmin)), dtype=Data_type)
    D_trn, dose_trn_max, dose_trn_min = NormalizeMinMax(D_trn)
    vec_trn = np.array(Vector(time_trn, x_data_trn, D_trn, L_trn, R_trn, P_trn), dtype=Data_type) 
       
    x_data_tst = NormalizeMinMaxTest(x_tst, zmax, zmin)
    y_tst = np.array(ShapeY(NormalizeMinMaxTest(y_tst, zmax, zmin)), dtype=Data_type)
    D_tst = NormalizeMinMaxTest(D_tst, dose_trn_max, dose_trn_min)
    vec_tst = np.array(Vector(time_tst, x_data_tst, D_tst, L_tst, R_tst, P_tst), dtype=Data_type)

    return vec_trn, vec_tst, y_trn, y_tst, time_trn, time_tst  


def time_weighting_root_mean_squared_error(y_truth, y_predict, timee):
    weights_list1 = list()
    weights1 = np.arange(0, 1.0, 0.03125)
    for i in range(len(timee)):
        weights_list1.append(weights1)
    rmse = np.sqrt(np.mean(weights_list1 * ((y_predict - y_truth)**2)) / np.sum(weights_list1)) 
    print('time weighting rmse: ', round(rmse,5))


start = time.time()
path = '/home/.../Data'
folder = "/home/.../"
percentage = 0.5
train_pc, valid_pc, test_pc = 0.7, 0.1, 0.2
Data_type = object 

data, metadata, names = Dataset(path) 
new_data = Level_min(data)
data_trn, data_tst, trn_name, tst_name, metadata_trn, metadata_tst = SplitTrainValidTest(new_data, metadata, names, train_pc, valid_pc)
time_trn, x_trn, y_trn, D_trn, L_trn, R_trn, P_trn, cant_trn, time_y_trn = Data(data_trn, metadata_trn, percentage)         
time_tst, x_tst, y_tst, D_tst, L_tst, R_tst, P_tst, cant_tst, time_y_tst = Data(data_tst, metadata_tst, percentage)

x_trn_1, x_trn_2, x_trn_3, x_trn_4 = divide(x_trn)
y_trn_1, y_trn_2, y_trn_3, y_trn_4 = divide(y_trn)
dose_trn_1, dose_trn_2, dose_trn_3, dose_trn_4 = divide(D_trn)
ligand_trn_1, ligand_trn_2, ligand_trn_3, ligand_trn_4 = divide(L_trn)
receptor_trn_1, receptor_trn_2, receptor_trn_3, receptor_trn_4 = divide(R_trn)
perturbation_trn_1, perturbation_trn_2, perturbation_trn_3, perturbation_trn_4 = divide(P_trn)
time_trn_1, time_trn_2, time_trn_3, time_trn_4 = divide(time_trn)

vec_x_1, vec_x_2, vec_x_3, vec_x_4, vec_x_5 = concatenation(x_trn_1, x_trn_2, x_trn_3, x_trn_4, x_tst)
vec_y_1, vec_y_2, vec_y_3, vec_y_4, vec_y_5 = concatenation(y_trn_1, y_trn_2, y_trn_3, y_trn_4, y_tst)
vec_dose_1, vec_dose_2, vec_dose_3, vec_dose_4, vec_dose_5 = concatenation(dose_trn_1, dose_trn_2, dose_trn_3, dose_trn_4, D_tst)
vec_lig_1, vec_lig_2, vec_lig_3, vec_lig_4, vec_lig_5 = concatenation(ligand_trn_1, ligand_trn_2, ligand_trn_3, ligand_trn_4, L_tst)
vec_rec_1, vec_rec_2, vec_rec_3, vec_rec_4, vec_rec_5 = concatenation(receptor_trn_1, receptor_trn_2, receptor_trn_3, receptor_trn_4, R_tst)
vec_per_1, vec_per_2, vec_per_3, vec_per_4, vec_per_5 = concatenation(perturbation_trn_1, perturbation_trn_2, perturbation_trn_3, perturbation_trn_4, P_tst)
vec_time_1, vec_time_2, vec_time_3, vec_time_4, vec_time_5 = concatenation(time_trn_1, time_trn_2, time_trn_3, time_trn_4, time_tst)

vec_trn_1, vec_tst_1, vec_y_trn_1, vec_y_tst_1, vec_time_trn_1, vec_time_tst_1 = normal(vec_x_1, vec_y_1, vec_time_1, vec_dose_1, vec_lig_1, vec_rec_1, vec_per_1, x_trn_1, y_trn_1, time_trn_1, dose_trn_1, ligand_trn_1, receptor_trn_1, perturbation_trn_1)
vec_trn_2, vec_tst_2, vec_y_trn_2, vec_y_tst_2, vec_time_trn_2, vec_time_tst_2 = normal(vec_x_2, vec_y_2, vec_time_2, vec_dose_2, vec_lig_2, vec_rec_2, vec_per_2, x_trn_2, y_trn_2, time_trn_2, dose_trn_2, ligand_trn_2, receptor_trn_2, perturbation_trn_2)
vec_trn_3, vec_tst_3, vec_y_trn_3, vec_y_tst_3, vec_time_trn_3, vec_time_tst_3 = normal(vec_x_3, vec_y_3, vec_time_3, vec_dose_3, vec_lig_3, vec_rec_3, vec_per_3, x_trn_1, y_trn_3, time_trn_3, dose_trn_3, ligand_trn_3, receptor_trn_3, perturbation_trn_3)
vec_trn_4, vec_tst_4, vec_y_trn_4, vec_y_tst_4, vec_time_trn_4, vec_time_tst_4 = normal(vec_x_4, vec_y_4, vec_time_4, vec_dose_4, vec_lig_4, vec_rec_4, vec_per_4, x_trn_4, y_trn_4, time_trn_4, dose_trn_4, ligand_trn_4, receptor_trn_4, perturbation_trn_4)
vec_trn_5, vec_tst_5, vec_y_trn_5, vec_y_tst_5, vec_time_trn_5, vec_time_tst_5 = normal(vec_x_5, vec_y_5, vec_time_5, vec_dose_5, vec_lig_5, vec_rec_5, vec_per_5, x_tst, y_tst, time_tst, D_tst, L_tst, R_tst, P_tst)

regressor = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, random_state=0)
#regressor = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=1, random_state=0)
#regressor = xgb.XGBRegressor(max_depth=3, n_estimators=10, learning_rate=0.5)

Ypred_trn1, Ypred_tst1, regressor1, mae_trn1, mae_tst1 = metrics_mae_auc_rmse(regressor, vec_trn_1, vec_y_trn_1, vec_tst_1, vec_y_tst_1, vec_time_trn_1, vec_time_tst_1)
Ypred_trn2, Ypred_tst2, regressor2, mae_trn2, mae_tst2 = metrics_mae_auc_rmse(regressor, vec_trn_2, vec_y_trn_2, vec_tst_2, vec_y_tst_2, vec_time_trn_2, vec_time_tst_2)
Ypred_trn3, Ypred_tst3, regressor3, mae_trn3, mae_tst3 = metrics_mae_auc_rmse(regressor, vec_trn_3, vec_y_trn_3, vec_tst_3, vec_y_tst_3, vec_time_trn_3, vec_time_tst_3)
Ypred_trn4, Ypred_tst4, regressor4, mae_trn4, mae_tst4 = metrics_mae_auc_rmse(regressor, vec_trn_4, vec_y_trn_4, vec_tst_4, vec_y_tst_4, vec_time_trn_4, vec_time_tst_4)
Ypred_trn5, Ypred_tst5, regressor5, mae_trn5, mae_tst5 = metrics_mae_auc_rmse(regressor, vec_trn_5, vec_y_trn_5, vec_tst_5, vec_y_tst_5, vec_time_trn_5, vec_time_tst_5)

end = time.time()
print('Time: ', end-start)

print('Average MAE Train: ', (mae_trn1 + mae_trn2 + mae_trn3 + mae_trn4 + mae_trn5) / 5)
print('Average MAE Test: ', (mae_tst1 + mae_tst2 + mae_tst3 + mae_tst4 + mae_tst5) / 5)

