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

encoder = OneHotEncoder(sparse_output=False)
one_hot_manure_type = encoder.fit_transform(manure_type)
print(one_hot_manure_type.shape)

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
# E, PPI = SPA_phase_one(data_absorbance_snv, 396, 9)
# print(PPI)
# input_data = absorbance
# input_data = E
input_data = signal.savgol_filter(absorbance, window_length=11, polyorder=3, deriv=2, mode='nearest')
# input_data = snv.fit_transform(input_data)
# input_data = np.concatenate((input_data, one_hot_manure_type), axis=1)
########################################
########################################

X_train, X_test, y_train, y_test = train_test_split(input_data, chemical_decom, test_size=.25)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25)

pls2 = PLSRegression(n_components=5)
pls2.fit(X_train, y_train)
y_pred_train = pls2.predict(X_train)
y_pred_test = pls2.predict(X_test)

msec = mean_squared_error(y_train, y_pred_train)
rmsec = np.sqrt(msec)

mset = mean_squared_error(y_test, y_pred_test)
rmset = np.sqrt(mset)

y_test = y_test.astype(float)
y_pred_test = y_pred_test.astype(float)

def calculate_rpd(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_ref = np.std(y_true)
    rpd = std_ref / rmse

    return rpd

print("r2 score: ", r2_score(y_train, y_pred_train))
print("rmsec: ", rmsec)
print('----------------')
print("r2 score: ", r2_score(y_test, y_pred_test))
print("rmsev: ", rmset)
print("rpd: ", calculate_rpd(y_test, y_pred_test))

with plt.style.context('ggplot'):
    plt.scatter(y_test, y_pred_test, color='red')
    plt.plot(y_test, y_test, '-g', label='Expected regression line')
    coeffs_test = np.polyfit(y_test.flatten(), y_pred_test.flatten(), deg=1)
    y_regression_test = coeffs_test[0] * y_test + coeffs_test[1]
    plt.plot(y_test, y_regression_test, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()

plt.show()

########################################################
#    code để khảo sát các thông số tối ưu cho model    #
########################################################
# init_ind = []
# rmset_arr = []
# for i in range(0, absorbance.shape[1]):
#     E, PPI = SPA_phase_one(data_absorbance_snv, i, 13)
#     input_data = np.concatenate((E, one_hot_manure_type), axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(input_data, chemical_decom, test_size=.2)
#     pls2 = PLSRegression(n_components=7)
#     pls2.fit(X_train, y_train)
#     # y_pred_val = pls2.predict(X_val)
#     y_pred_test = pls2.predict(X_test)

#     # msev = mean_squared_error(y_val, y_pred_val)
#     # rmsev = np.sqrt(msev)

#     mset = mean_squared_error(y_test, y_pred_test)
#     rmset = np.sqrt(mset)
#     init_ind.append(i)
#     rmset_arr.append(rmset)
    
# init_ind = np.array(init_ind)
# rmset_arr = np.array(rmset_arr)
# plt.plot(init_ind, rmset_arr, marker='x')
# plt.xlabel('init index')
# plt.ylabel('rmset')
# plt.show()

# init_ind = []
# rmset_arr = []
# rpd_arr = []
# r2_arr = []
# for i in range(1, 15):
#     E, PPI = SPA_phase_one(data_absorbance_snv, 396, i)
#     input_data = np.concatenate((E, one_hot_manure_type), axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(input_data, chemical_decom, test_size=.2)
#     pls2 = PLSRegression(n_components=5)
#     pls2.fit(X_train, y_train)
#     # y_pred_val = pls2.predict(X_val)
#     y_pred_test = pls2.predict(X_test)

#     # msev = mean_squared_error(y_val, y_pred_val)
#     # rmsev = np.sqrt(msev)

#     mset = mean_squared_error(y_test, y_pred_test)
#     rmset = np.sqrt(mset)
#     rpd = calculate_rpd(y_test, y_pred_test)
#     r2 = r2_score(y_test, y_pred_test)
#     # init_ind.append(i)
#     rmset_arr.append(rmset)
#     rpd_arr.append(rpd)
#     r2_arr.append(r2)
#     print(i)
    
# # init_ind = np.array(init_ind)
# rmset_arr = np.array(rmset_arr)
# rpd_arr = np.array(rpd_arr)
# r2_arr = np.array(r2_arr)
# num_comps = np.arange(1, 15)

# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(121)
# ax.plot(num_comps, rmset_arr, marker='x', color='blue')
# ax.set_xlabel('K-selected')
# ax.set_ylabel('rmset')

# ax2 = fig.add_subplot(122)
# ax2.plot(num_comps, r2_arr, marker='s', color='red')
# ax2.set_xlabel('K-selected')
# ax2.set_ylabel('r2 score')

# plt.tight_layout()  
# plt.show()

# print(rmset_arr.argmin())
# print(rmset_arr[rmset_arr.argmin()])
# print(r2_arr[rmset_arr.argmin()])
# print(rpd_arr[rmset_arr.argmin()])
#######################################################
#######################################################

#########################
#    lazy regression    #
#########################
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
#########################
#########################