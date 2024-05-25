import os
import sys

import pandas as pd
import numpy as np
import timeit

import pywt
import scipy.io as spio
from scipy.stats import entropy
from collections import Counter

from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy import signal
from scipy.signal import lfilter,butter
dir = "D:\sem6\year_project\eeg_raw_data\eeg_raw_data\session1"
os.chdir(dir)
os.getcwd()

files = os.listdir("D:\sem6\year_project\eeg_raw_data\eeg_raw_data\session1")
print(files)
def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
participant_trial = []
features_table = pd.DataFrame(columns=range(2170))
for file in files:
  mat_file = spio.loadmat("D:\sem6\year_project\eeg_raw_data\eeg_raw_data\session1/" + file)
  keys = [key for key, values in mat_file.items() if key != '__header__' and key != '__version__' and key != '__globals__' ]
  for data_file in keys:
    data_df = pd.DataFrame(mat_file[data_file])
    #downsampling the signal(data frame)
    f=signal.decimate(data_df,5)
    #band pass filtering the signal between 1hz and 75hz       
    filtered_sig=butter_bandpass_filter(f,1.0,75.0,200.0)
    fs=200.0  #sampling frequency
    win = 4 *fs #4 seconds window 
    #power spectral density of the signal
    freqs, psd = signal.welch(filtered_sig, fs, nperseg=win)
    psd_df=pd.DataFrame(psd)
    #Feature extraction part
    int_psd=[]
    int_entropy=[]
    int_mean=[]
    int_var=[]
    int_max=[]
    int_min=[]
    int_std=[]
    for channel in psd_df.iterrows():  # Iterating through the 62 channels
        eeg_bands = {'Delta': (0, 4),
                             'Theta': (4, 8),
                             'Alpha': (8, 14),
                             'Beta': (14, 31),
                             'Gamma': (31, 50)}
        eeg_band_fft = dict()
        freq_band=dict() 
        entropy_band=dict()
        mean_band=dict()
        var_band=dict()
        max_band=dict()
        min_band=dict()
        std_band=dict()
        psd_data=channel[1]
        for band in eeg_bands:  # splliting into five frequency bands
            freq_ix = np.where((freqs >= eeg_bands[band][0]) & 
                                       (freqs <= eeg_bands[band][1]))[0]
            freq_band[band]=psd_data[freq_ix]
            #DE feature extraction of 5 frequency bands
            entropy_band[band]=np.linalg.norm(np.log10(psd_data[freq_ix])) 
            #PSD feature extraction of 5 frequency bands
            eeg_band_fft[band] = np.linalg.norm(psd_data[freq_ix],)
            #Mean feature extraction of 5 frequency bands
            mean_band[band]=np.mean(psd_data[freq_ix])
            #Varience feature extraction of 5 frequency bands
            var_band[band]=np.var(psd_data[freq_ix])
            #Maximum value feature extraction of 5 frequency bands
            max_band[band]=np.max(psd_data[freq_ix])
            #Minimum value feature extraction of 5 frequency bands
            min_band[band]=np.min(psd_data[freq_ix])
            #Standard deviation feature extraction of 5 frequency bands
            std_band[band]=np.std(psd_data[freq_ix])
        eeg_band_fft_1=list(eeg_band_fft.values())
        entropy_band_1=list(entropy_band.values())
        mean_band_1=list(mean_band.values())
        var_band_1=list(var_band.values())
        max_band_1=list(max_band.values())
        min_band_1=list(min_band.values())
        std_band_1=list(std_band.values())
        int_entropy.append(entropy_band_1)
        int_psd.append(eeg_band_fft_1)
        int_mean.append(mean_band_1)
        int_var.append(var_band_1)
        int_max.append(max_band_1)
        int_min.append(min_band_1)
        int_std.append(std_band_1)
    psd_features=[]
    entropy_features=[]
    mean_features=[]
    var_features=[]
    max_features=[]
    min_features=[]
    std_feautures=[]
    for i in range(len(int_psd)):
         for j in range(len(int_psd[0])):
             psd_features.append(int_psd[i][j])
             entropy_features.append(int_entropy[i][j])
             mean_features.append(int_mean[i][j])
             var_features.append(int_var[i][j])
             max_features.append(int_max[i][j])
             min_features.append(int_min[i][j])
             std_feautures.append(int_std[i][j])
    features=psd_features+entropy_features+mean_features+var_features+max_features+min_features+std_feautures        
    participant_trial.append(features)
    features_table.loc[len(features_table.index)] = features
        
  print(file) #printing mat files
y=pd.read_csv('D:\sem6\year_project\output_1.csv',header=None) #reading ouput csv file
X = features_table.iloc[:,:].values
Y = y.iloc[:, :].values

#splittng data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size = 0.2, random_state=42)

#Standardizes the data by subtracting the mean and then scaling to unit varience
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#classification using SVM
from sklearn.svm import SVC
import sklearn 
svclassifier = SVC(kernel='rbf',C=1000,gamma=0.0001)
svclassifier.fit(X_train_scaled, Y_train.ravel())  
Y_pred = svclassifier.predict(X_test_scaled) 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))                         
print('accuracy=',sklearn.metrics.accuracy_score(Y_test,Y_pred)*100)   
    
    

    
    
    