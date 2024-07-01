from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy import signal
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
from sklearn.preprocessing import OneHotEncoder
from kennard_stone import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
chemical_decom = chemical_decom.astype('float64')

encoder = OneHotEncoder(sparse_output=False)
one_hot_manure_type = encoder.fit_transform(manure_type)
print(type(one_hot_manure_type[0, 0]))

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

########################################
#    Code để khảo sát preprocessing    #
########################################     
E, PPI = SPA_phase_one(data_absorbance_snv, 396, 9)
# print(PPI)
# input_data = absorbance
input_data = E
# input_data = signal.savgol_filter(absorbance, window_length=11, polyorder=3, deriv=2, mode='nearest')
# input_data = snv.fit_transform(input_data)
input_data = np.concatenate((input_data, one_hot_manure_type), axis=1)
########################################
########################################

print(input_data.shape)
print(chemical_decom.shape)
X_train, X_temp, y_train, y_temp = train_test_split(input_data, chemical_decom, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_valid = X_valid.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
y_valid = y_valid.astype('float64')

scaler = StandardScaler()
y_train_normalized = scaler.fit_transform(y_train)
y_test_normalized = scaler.transform(y_test)
y_val_normalized = scaler.transform(y_valid)

y_train_normalized = y_train_normalized.reshape(-1, 1)
y_test_normalized = y_test_normalized.reshape(-1, 1)
y_val_normalized = y_val_normalized.reshape(-1, 1)


clf = TabNetRegressor(
    verbose=1
)

clf.fit(
    X_train=X_train, y_train=y_train_normalized,
    eval_set=[(X_train, y_train_normalized), (X_valid, y_val_normalized)],
    eval_name=['train', 'valid'],
    eval_metric=['mse'],
    batch_size=32, virtual_batch_size=16,
    max_epochs=100,
    patience=100,
    num_workers=0,
    drop_last=False,
)
preds = clf.predict(X_test)
preds = scaler.inverse_transform(preds)
print(preds)
y_true = y_test
test_score = mean_squared_error(y_pred=preds, y_true=y_true)
print(f"BEST VALID SCORE FOR: {clf.best_cost}")
print(f"FINAL TEST SCORE FOR: {test_score}")

def calculate_rpd(y_true, y_pred):
    calculate_rpd.__name__ = 'RPD'
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_ref = np.std(y_true)
    rpd = std_ref / rmse

    return rpd

print("rpd: ", calculate_rpd(y_test, preds))

with plt.style.context('ggplot'):
    plt.scatter(y_test, preds, color='red')
    plt.plot(y_test, y_test, '-g', label='Expected regression line')
    coeffs_test = np.polyfit(y_test.flatten(), preds.flatten(), deg=1)
    y_regression_test = coeffs_test[0] * y_test + coeffs_test[1]
    plt.plot(y_test, y_regression_test, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()

plt.show()