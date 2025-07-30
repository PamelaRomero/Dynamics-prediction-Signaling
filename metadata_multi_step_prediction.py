from sklearn.preprocessing import OneHotEncoder



def OHotEncoder(X, list_1):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    enc.categories_
    list_2 = list()
    for i in range(len(list_1)):
        list_2.append([list_1[i]])
    result = enc.transform(list_2).toarray()

    list_enc1 = list()
    list_enc2 = list()
    for j in range(len(result)):
        for k in range(len(result[j])):
            list_enc1.append(result[j][k])
        list_enc2.append(list_enc1)
        list_enc1 = []
    return list_enc2


def Dose(info):
    name = list()
    list_enc_only_dose = list()
    list_enc_only_dose_number  = list()
    list_enc_number = list()
    for t in range(len(info)):
        name.append(info[t])
        spliit = info[t].split()
        for i in range(len(spliit)):
            if 'nM' in spliit[i]:
                list_enc_only_dose.append(spliit[i])
    
    for r in range(len(list_enc_only_dose)):
        for k in range(len(list_enc_only_dose[r])):
            if list_enc_only_dose[r][k] != "n":
                if list_enc_only_dose[r][k] == ",":
                    list_enc_only_dose_number.append(".")
                else:
                    list_enc_only_dose_number.append(list_enc_only_dose[r][k])
            else:
                break
        list_enc_number.append(float("".join(list_enc_only_dose_number))) 
        list_enc_only_dose_number = []

    return list_enc_number, name


def LigandOHE(info):
    list_ligand = list()
    for t in range(len(info)):
        spliit = info[t].split()
        if 'hCG' in spliit:
            list_ligand.append('hCG')
        elif 'LH' in spliit:
            list_ligand.append('LH')
        elif 'FSH' in spliit:
            list_ligand.append('FSH')
        else:
            list_ligand.append('None')
    
    X = [['hCG'], ['LH'], ['FSH'], ['None']]
    list_enc = OHotEncoder(X, list_ligand)
    return list_enc


def ReceptorOHE(info):
    list_receptor = list()
    for t in range(len(info)):
        spliit = info[t].split()
        if 'LHR' in spliit and 'WT' in spliit:
            list_receptor.append('hLHR') 
        elif 'RLH' in spliit:
            list_receptor.append('hLHR') 
        elif 'hLHR' in spliit:
            list_receptor.append('hLHR') 
        elif 'LHR-T' in spliit:
            list_receptor.append('LHR-T')
        elif 'mLHR' in spliit:
            list_receptor.append('mLHR') 
        elif 'hFSHR' in spliit:
            list_receptor.append('hFSHR') 
        elif 'mFSHR' in spliit:
            list_receptor.append('mFSHR') 
        elif 'FSHR' in spliit:
            list_receptor.append('hFSHR') 
        elif 'LHR' in spliit:
            list_receptor.append('hLHR') 
        else:
            list_receptor.append('None')

    X = [['hLHR'], ['LHR_T'], ['mLHR'], ['hFSHR'], ['mFSHR'], ['None']]
    list_enc = OHotEncoder(X, list_receptor)
    return list_enc 


def PerturbationOHE(info):
    list_perturbation = list()
    for t in range(len(info)):
        spliit = info[t].split()
        if 'Dyngo4A' in spliit:
            list_perturbation.append('Dyngo4A')
        elif 'PitStop' in spliit:
            list_perturbation.append('PitStop')
        elif 'Es9-17' in spliit:
            list_perturbation.append('Es9_17')
        elif 'YM254890' in spliit:
            list_perturbation.append('YM254890')
        elif 'F8' in spliit:
            list_perturbation.append('F8')
        elif 'Nb37' in spliit and 'NES' in spliit:
            list_perturbation.append('Nb37')
        elif 'Nb37' in spliit and 'CAAX' in spliit:
            list_perturbation.append('Nb37')
        elif 'Nb37' in spliit  and '2xFYVE' in spliit:
           list_perturbation.append('Nb37')
        elif 'mCherry' in spliit:
            list_perturbation.append('mCherry')
        elif 'Control' in spliit:
            list_perturbation.append('Control')
        else:
            list_perturbation.append('None')

    X = [['Dyngo4A'], ['PitStop'], ['Es9_17'], ['YM254890'], ['F8'], ['Nb37'], ['mCherry'], ['Control'], ['None']]
    list_enc = OHotEncoder(X, list_perturbation)
    return list_enc