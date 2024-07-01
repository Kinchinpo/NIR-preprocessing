import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy import signal
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from kennard_stone import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from linearboost import LinearBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

file_path_nir = r'dataverse_files/spectra-1.csv'
df = pd.read_csv(file_path_nir)

wavelength = df.columns[1:].to_numpy()
wavelength = np.array([float(item[:-3]) if item.endswith('_nm') else item for item in wavelength])
absorbance = df.values[:][0:]
absorbance = np.delete(absorbance, 0, axis=1)
absorbance = np.vectorize(lambda x: float(x.replace(',', '.')))(absorbance)

print(wavelength[396])
file_path_manure = r'dataverse_files/chemical_analysis.xlsx'
manure_data = pd.read_excel(file_path_manure)
manure_data = manure_data.values
manure_data = manure_data[:, 3:]
# print(manure_data)
# print(manure_data.shape)
manure_type = manure_data[:, :1]
chemical_decom = manure_data[:, 5:6]

le = LabelEncoder()
manure_type_encoded = le.fit_transform(manure_type)
manure_type_encoded[manure_type_encoded != 0] = 1
# print(manure_type_encoded)

sg_smooth = signal.savgol_filter(absorbance, window_length=7, polyorder=2, deriv=0, mode='nearest')
snv = StandardNormalVariate()
data_absorbance_snv = snv.fit_transform(sg_smooth)
msc = MultiplicativeScatterCorrection(use_mean=True)
data_absorbance_msc = msc.fit_transform(sg_smooth)

def SPA_phase_one(data, init_index, K):
    ENDMEM = np.zeros((data.shape[0], K))
    PPI = np.zeros(K)
    PPI = np.int64(PPI)
    
    ak = data[:, init_index:init_index + 1]
    eyeMat = np.eye(data.shape[0])
    ak_Mat = np.dot(ak, ak.transpose())
    ak_pow = linalg.norm(ak, 2) ** 2
    SPA_ProjMat = (eyeMat - ak_Mat / ak_pow)
    PPI[0] = init_index
    ENDMEM[:, 0:1] = data[:, init_index:init_index + 1]
    
    for i in range(1, K):
        normSquareArray = np.zeros(data.shape[1])
        for j in range(0, data.shape[1]):
            normSquareArray[j] = linalg.norm(np.dot(SPA_ProjMat, data[:, j:j + 1]), 2)
        
        PPI[i] = normSquareArray.argmax()
        ENDMEM[:, i:i + 1] = data[:, PPI[i]:PPI[i] + 1]
        ak          = np.dot(SPA_ProjMat, ENDMEM[:, i:i + 1])
        eyeMat      = np.eye(data.shape[0])
        ak_Mat      = np.dot(ak, ak.transpose())
        ak_pow      = linalg.norm(ak, 2) ** 2
        SPA_ProjMat = np.dot((eyeMat - ak_Mat / ak_pow), SPA_ProjMat)

    return ENDMEM, PPI

E, PPI = SPA_phase_one(data_absorbance_snv, 396, 9)
# print(PPI)
# input_data = absorbance
input_data = E
# input_data = signal.savgol_filter(absorbance, window_length=11, polyorder=2, deriv=1, mode='nearest')
# input_data = snv.fit_transform(input_data)
# input_data = np.concatenate((input_data, one_hot_manure_type), axis=1)
X_train, X_test, y_train, y_test = train_test_split(input_data, manure_type_encoded, test_size=.25)

linearboost_estimator = LinearBoostClassifier()
linearboost_estimator.fit(X_train, y_train)
y_pred = linearboost_estimator.predict(X_test)
print(y_test)
print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Precision, Recall, F1 score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}')
