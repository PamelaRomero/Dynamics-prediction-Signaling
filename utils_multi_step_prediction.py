import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from metadata_multi_step_prediction import Dose, LigandOHE, ReceptorOHE, PerturbationOHE



def Dataset(path):
    time_list, signal_list, names = Main(path)
    dose_list, name_list = Dose(names)
    LigandOHE_list = LigandOHE(names)
    ReceptorOHE_list = ReceptorOHE(names)
    PerturbationOHE_list = PerturbationOHE(names)
    data, metadata = Data_Metadata(time_list, signal_list, dose_list, LigandOHE_list, ReceptorOHE_list, PerturbationOHE_list)

    print('dose_list: ', len(dose_list))
    print('LigandOHE_list: ', len(LigandOHE_list))
    print('ReceptorOHE_list: ', len(ReceptorOHE_list))
    print('PerturbationOHE_list: ', len(PerturbationOHE_list))

    return data, metadata, name_list


def Main(path):
    list_path = paths(path)
    names, signal, signal_list, time, time_list = list(), list(), list(), list(), list()
    count = 0
   
    for i in range(len(list_path)):
        file_on = list_path[i]
        file_read = pd.ExcelFile(file_on)
        sheet_names = file_read.sheet_names
        df = pd.read_excel(file_on, sheet_names[0])

        rows, columns = df.shape
        row_Time, col_Time = InitialRowColumn(df)
        for j in range(col_Time+1, columns):
            for k in range(row_Time, rows): 
                if (k > row_Time):
                    signal.append(float(df.iloc[k, j]))  
                    time.append(float(df.iloc[k, col_Time]))
                else:
                    if count == 0:
                        names.append(df.iloc[k, j])
                        count = count + 1
                    elif count == 1:
                        names.append(names[-1])
                        count = count + 1
                    else:
                        names.append(names[-1])
                        count = 0
            signal_list.append(signal)
            time_list.append(time)
            signal = []
            time = []
   
    return time_list, signal_list, names


def paths(path):    
    files = os.listdir(path)  
    list_path = list()
    for file in files:         
        files2 = os.listdir(path + '/' + file)
        for file2 in files2:
            if '.xlsx' not in file2:
                files3 = os.listdir(path + '/' + file + '/' + file2)
                for file3 in files3:
                    list_path.append(path + '/' + file + '/' + file2 + '/' + file3)
            else:
                list_path.append(path + '/' + file + '/' + file2)
    return list_path


def InitialRowColumn(df):
    rows, columns = df.shape
    for k in range(0, rows): 
        for l in range(0, columns):
            if(df.iloc[k,l] == 'Induced BRET') or (df.iloc[k,l] == 'Induced BRET normalized on baseline'):
                k2, l2 = k, l
            else:
                pass

    for k1 in range(k2, rows): 
        for l1 in range(l2, columns):
            if (df.iloc[k1,l1] == 'Time (min)')  or (df.iloc[k1,l1] == 'Time WASH (min)') or (df.iloc[k1,l1]) == 'Time NO WASH (min)' or (df.iloc[k1,l1] == 'Time (min) for WASH') or (df.iloc[k1,l1] == 'Time NO WASH (min)'):  
                row_Time, col_Time = k1, l1
                break                
            else:
                pass
    return row_Time, col_Time


def Data_Metadata(time_list, signal_list, dose_list, ligand_list, receptor_list, perturbation_list):
    metadata = list()
    data = list()
    for i in range(len(dose_list)):
        metadata.append(dict(dose=dose_list[i], ligand=ligand_list[i], receptor=receptor_list[i], perturbation=perturbation_list[i]))
    metadata = pd.DataFrame(metadata)
    for j in range(len(signal_list)):
        data.append(np.array([time_list[j], signal_list[j]]))
    return data, metadata


def Level_min(data):
    lenn = list()
    for i in range(len(data)):
        length = len(data[i][0])
        lenn.append(length)
    min_lista = min(lenn)
    new_data = list()
    for j in range(len(data)):
        length2 = len(data[j][0])
        if length2 != min_lista:
            new_data.append([data[j][0][:min_lista], data[j][1][:min_lista]])
        else:
            new_data.append([data[j][0][:min_lista], data[j][1][:min_lista]])
    return new_data


def SplitTrainValidTest(data, metadata, name_lista, trn_pc, val_pc):
    new_data = int(len(data) / 3)
    to_trn = round(int(new_data * trn_pc) * 3)
    to_val = round(int(new_data * (trn_pc + val_pc)) * 3)

    trn_data = data[:to_trn]
    val_data = data[to_trn:to_val]
    tst_data = data[to_val:]

    print('trn_data: ',len(trn_data))
    print('val_data: ', len(val_data))
    print('tst_data: ', len(tst_data))

    trn_name = EveryThree(name_lista[:to_trn])
    val_name = EveryThree(name_lista[to_trn:to_val])
    tst_name = EveryThree(name_lista[to_val:])

    trn_metadata = metadata[:to_trn]
    val_metadata = metadata[to_trn:to_val]
    tst_metadata = metadata[to_val:]
    
    listData_trn, metaData_trn = Dataa(trn_data, trn_metadata)
    listData_val, metaData_val = Dataa(val_data, val_metadata)
    listData_tst, metaData_tst = Dataa(tst_data, tst_metadata)
    
    return listData_trn, listData_val, listData_tst, trn_name, val_name, tst_name, metaData_trn, metaData_val, metaData_tst
        

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


def Data(datos, metadatos, c):
    res = metadatos['receptor'].to_list()
    lig = metadatos['ligand'].to_list()
    dos = metadatos['dose'].to_list()
    per = metadatos['perturbation'].to_list()

    BRET_Y, BRET, TIME, DOSE, LIGAND, RECEPTOR, PERTURBATION, cantidad, TIME_Y = list(), list(), list(), list(), list(), list(), list(), list(), list()  
    
    for i in range(len(datos)):
        t, x = datos[i][0], datos[i][1]

        cant = int(round(c * len(x)))
        BRET_Y.append(x[cant:].tolist())

        DOSE.append(dos[i])
        LIGAND.append(lig[i])
        RECEPTOR.append(res[i])
        PERTURBATION.append(per[i])

        cantidad.append(cant)
    
        BRET.append(x[:cant].tolist())
        TIME.append(t[:cant].tolist())
        TIME_Y.append(t[cant:].tolist())       
       
    return TIME, BRET, BRET_Y, DOSE, LIGAND, RECEPTOR, PERTURBATION, cantidad, TIME_Y


def Normalization(x_data_trn, y_data_trn):
    x_data_trn1, x_data_trn_max, x_data_trn_min = NormalizeMinMax(x_data_trn)
    y_data_trn1, y_data_trn_max, y_data_trn_min = NormalizeMinMax(y_data_trn)

    lis = [x_data_trn_max, x_data_trn_min, y_data_trn_max, y_data_trn_min]
    z_min = min(lis)
    z_max = max(lis)

    return z_min, z_max


def NormalizeMinMaxTest(x, maax, miin):
    x_new = list()
    for j in range(len(x)):
        x_new.append((x[j] - miin) / (maax - miin))
    return x_new


def NormalizeMinMax(x_data_train):
    ma_lista = list()
    mi_lista = list()
    x_new = list()
    for i in range(len(x_data_train)):
        ma_lista.append(np.max(x_data_train[i]))  
        mi_lista.append(np.min(x_data_train[i]))

    maa = np.max(ma_lista)
    mii = np.min(mi_lista)

    for j in range(len(x_data_train)):
        x_new.append((x_data_train[j] - mii) / (maa - mii))
    return x_new, maa, mii


def ShapeY(vector):
    l1, l2 = list(), list()
    for i in range(len(vector)):
        for j in range(len(vector[i])):
            l1.append((vector[i][j].tolist()))
        l2.append(l1)
        l1 = []
    return l2  


def ShapeV(vector):
    l1, l2 = list(), list()
    for i in range(len(vector)):
        for j in range(len(vector[i])):
            l1.append((vector[i][j]))
        l2.append(l1)
        l1 = []
    return l2 


def Vector(time, x_data, dose, ligand, receptor, perturbation):
    vec, vector_list, x_list, time_list, meta = list(), list(), list(), list(), list()
    for k in range(len(x_data)):
        meta = [dose[k], ligand[k][0],ligand[k][1],ligand[k][2],ligand[k][3],receptor[k][0],receptor[k][1],receptor[k][2],receptor[k][3],receptor[k][4],receptor[k][5], perturbation[k][0],perturbation[k][1],perturbation[k][2],perturbation[k][3],perturbation[k][4],perturbation[k][5],perturbation[k][6],perturbation[k][7],perturbation[k][8]]        
        for xd in range(len(x_data[k])):
            x_list.append(x_data[k][xd])
            time_list.append(time[k][xd])
        vec = list(itertools.chain(meta, time_list, x_list))
        x_list = []
        time_list = []
        meta = []

        vector_list.append(vec)
        vec = []       
            
    return ShapeV(vector_list)


def GridSearch(vec_trn, y_trn):
    params_grid = {
        'max_depth': [1, 3, 5, 10, 15],
        'min_samples_leaf': [1, 3, 5, 10, 15],
        'n_estimators': [1, 5, 10] 
    }
    #regre = DecisionTreeRegressor(random_state=0)
    regre = RandomForestRegressor(random_state=0)
    regressor = GridSearchCV(estimator=regre, param_grid=params_grid, cv=5, scoring='neg_mean_squared_error')
    regressor.fit(vec_trn, y_trn)
    print("-------------- GridSearchCV --------------")
    print(regressor.best_params_)
    print("------------------------------------------")


def Metrics(gt, pred):
    MAE(gt, pred)
    

def MAE(gt, pred):
    lista = list()
    for i in range(len(gt)):
        lista.append(np.mean(np.abs([e1 - e2 for e1, e2 in zip(gt[i],pred[i])])))
    print('Mean Absolute Error (MAE):', round(sum(lista)/len(lista),4))


def Graph(y, time, time_y, x, Ypred, name_list, folder_name):
    for p in range(len(y)):
        plt.figure()
        plt.plot(time[p], x[p], color = 'red', label='Ground Truth')
        plt.plot([time[p][-1], time_y[p][0]], [x[p][-1], y[p][0]], color = 'red')
        plt.plot(time_y[p], y[p], color = 'red')
        plt.plot(time_y[p], Ypred[p], color = 'blue', label='Random Forest Regressor')  

        plt.legend()
        plt.title('{}'.format(name_list[p]))
        plt.ylabel('BRET Ratio')
        plt.xlabel('Time')
        plt.savefig("{}/fig_{}.png".format(folder_name, p), dpi=200)
        plt.close()  


def area_under_curve(y_tst, Ypred_tst, time_y_tst):
    auc_pred_total = 0
    auc_truth_total = 0

    auc_pred_total_trapz = 0
    auc_truth_total_trapz = 0

    auc_pred_total_simpson = 0
    auc_truth_total_simpson = 0

    for i in range(len(time_y_tst)):
        # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        auc_pred = round(auc(time_y_tst[i], Ypred_tst[i]))
        auc_truth = round(auc(time_y_tst[i], y_tst[i]))
        diff_auc = np.abs(auc_truth - auc_pred)
        
        # REFERENCE: https://numpy.org/devdocs/reference/generated/numpy.trapezoid.html
        auc_pred_trapz = round(np.trapezoid(Ypred_tst[i], time_y_tst[i], dx=1))
        auc_truth_trapz = round(np.trapezoid(y_tst[i], time_y_tst[i], dx=1))
        diff_trapz = round(np.abs(auc_truth_trapz - auc_pred_trapz))

        # REFERENCIA: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html
        auc_pred_simpson = round(integrate.simpson(Ypred_tst[i], time_y_tst[i]))
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


def time_weighting_root_mean_squared_error(y_truth, y_predict, timee):
    weights_list = list()
    weights = np.arange(0, 1.0, 0.03125)
    for i in range(len(timee)):
        weights_list.append(weights)
    rmse = np.sqrt(np.mean(weights_list * ((y_predict - y_truth)**2)) / np.sum(weights_list)) 
    print('Time-weighting rmse: ', round(rmse,5))
