import itertools
import pandas as pd
import numpy as np

from sys import argv
from scipy.fftpack import fft, ifft
from sklearn.metrics import confusion_matrix, roc_curve
from tslearn.clustering import TimeSeriesKMeans

EPS = 1e-8

def average_filter(values, n=3):
    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


kVal = 300
if "-k" in argv:
    kVal = int(argv[argv.index("-k") + 1])
    print('K value = ' + str(kVal))

distanceMatrix = 'eucl'
if "-dtw" in argv:
    distanceMatrix = 'dtw'
print('Distance matric = ' + distanceMatrix)

train_windows = pd.read_csv('train-forecasts.csv', index_col=False)
test_windows = pd.read_csv('test-forecasts.csv', index_col=False)

train_data_without_attacks = train_windows.loc[train_windows['percentage'] == 1]
train_data_without_attacks = train_data_without_attacks.drop(['result', 'percentage', 'slot', 'duration'], 1)

test_data = test_windows.drop(['result', 'percentage', 'slot', 'duration'], 1)
test_data = test_data.reset_index(drop=True)
test_result = test_windows['result']

if distanceMatrix == 'eucl':
    model = TimeSeriesKMeans(n_clusters=kVal, n_init=10).fit(train_data_without_attacks.values)
elif distanceMatrix == 'dtw':
    model = TimeSeriesKMeans(n_clusters=kVal, metric='dtw', n_init=10).fit(train_data_without_attacks.values)

df = pd.DataFrame()
df['result'] = test_result
df['sr_value'] = -1
df['prediction'] = -1

for i in range(len(test_data)):
    pred = model.predict([test_data.loc[i].values])[0]
    closest_centroid = list(itertools.chain(*model.cluster_centers_[pred]))

    residual = test_data.loc[i].values - closest_centroid
    trans = fft(residual)
    magnitudes = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(magnitudes <= EPS)[0]
    magnitudes[eps_index] = EPS

    mag_log = np.log(magnitudes)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - average_filter(mag_log, n=48))

    trans.real = trans.real * spectral / magnitudes
    trans.imag = trans.imag * spectral / magnitudes
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = ifft(trans)
    saliency_map = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

    df.loc[i, 'sr_value'] = max(saliency_map)

fpr, tpr, thresholds = roc_curve(df['result'], df['sr_value'])
threshold = pd.Series(tpr-fpr, index=thresholds, name='tf').idxmax()
df['prediction'] = df['sr_value'].map(lambda x: 1 if x > threshold else 0)

tn, fp, fn, tp = confusion_matrix(df['result'], df['prediction']).ravel()
precision = tp / (tp + fp)
accuracy = (tp + tn) / len(df)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
fpr = fp / (fp + tn)

print('Accuracy = ' + str(accuracy))
print('Precision = ' + str(precision))
print('Recall = ' + str(recall))
print('F1 = ' + str(f1))
print('FPR = ' + str(fpr))

