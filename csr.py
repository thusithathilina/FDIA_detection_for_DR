import itertools
import pandas as pd
import numpy as np
import cmath

from sys import argv
from scipy.fftpack import fft, ifft
from sklearn.metrics import confusion_matrix, roc_curve
from tslearn.clustering import TimeSeriesKMeans

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

q = 15
for i in range(len(test_data)):
    pred = model.predict([test_data.loc[i].values])[0]
    closest_centroid = list(itertools.chain(*model.cluster_centers_[pred]))

    residual = test_data.loc[i].values - closest_centroid
    fft_residual = fft(residual)
    amplitude = np.abs(fft_residual)
    phase = np.angle(fft_residual)
    log = np.log(amplitude)

    average_log = np.convolve(log, [1 / (q * q)] * q, 'same')
    spectral_residual = log - average_log
    saliency_map = np.abs(ifft(np.exp(spectral_residual + phase * cmath.sqrt(-1))))

    df.loc[i, 'sr_value'] = max(saliency_map)

fpr, tpr, thresholds = roc_curve(df['result'], df['sr_value'])
threshold = pd.Series(tpr-fpr, index=thresholds, name='tf').idxmax()
df['prediction'] = df['sr_value'].map(lambda x: 1 if x > threshold else 0)

tn, fp, fn, tp = confusion_matrix(df['result'], df['prediction']).ravel()
precision = tp / (tp + fp)
accuracy = (tp + tn) / len(df)
recall = tp / (tp + tn)
f1 = 2 * (precision * recall) / (precision + recall)
fpr = fp / (fp + tn)

print('Accuracy = ' + str(accuracy))
print('Precision = ' + str(precision))
print('Recall = ' + str(recall))
print('F1 = ' + str(f1))
print('FPR = ' + str(fpr))


