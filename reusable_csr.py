import cmath
import itertools

import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from sklearn.metrics import confusion_matrix, roc_curve
from tslearn.clustering import TimeSeriesKMeans


def train_model(train_data, k=300, distance_matrix='eucl'):
    if distance_matrix == 'dtw':
        model = TimeSeriesKMeans(n_clusters=k, metric='dtw', n_init=10).fit(train_data)
    else:
        model = TimeSeriesKMeans(n_clusters=k, n_init=10).fit(train_data)
    return model


def save_model(model, path):
    model.to_hdf5(path)


def load_model(path):
    return TimeSeriesKMeans().from_hdf5(path)


def get_saliency_map(residual_forecast, q=15):
    fft_residual = fft(residual_forecast)
    amplitude = np.abs(fft_residual)
    phase = np.angle(fft_residual)
    log = np.log(amplitude)

    average_log = np.convolve(log, [1 / (q * q)] * q, 'same')
    spectral_residual = log - average_log
    saliency_map = np.abs(ifft(np.exp(spectral_residual + phase * cmath.sqrt(-1))))
    return saliency_map


if __name__ == "__main__":
    mode = 'TRAIN'
    # mode = 'LOAD'

    if mode == 'TRAIN':
        train_windows = pd.read_csv('train-forecasts.csv', index_col=False).head(100)

        # filter the attack-free data
        train_data_without_attacks = train_windows.loc[train_windows['percentage'] == 1]
        # drop unwanted columns
        train_data_without_attacks = train_data_without_attacks.drop(['result', 'percentage', 'slot', 'duration'], 1)

        model = train_model(train_data_without_attacks.values)
        save_model(model, 'sample_model.hdf5')
    else:
        model = load_model('sample_model.hdf5')

    test_windows = pd.read_csv('test-forecasts.csv', index_col=False).head(50)
    # drop unwanted columns
    test_data = test_windows.drop(['result', 'percentage', 'slot', 'duration'], 1)
    test_data = test_data.reset_index(drop=True)
    test_result = test_windows['result']

    df = pd.DataFrame()
    df['result'] = test_result
    df['sr_value'] = -1
    df['prediction'] = -1


    for i in range(len(test_data)):
        pred = model.predict([test_data.loc[i].values])[0]
        closest_centroid = list(itertools.chain(*model.cluster_centers_[pred]))
        s_map = get_saliency_map(test_data.loc[i].values - closest_centroid)
        df.loc[i, 'sr_value'] = max(s_map)

    #calculate the optimal threshold using
    fpr, tpr, thresholds = roc_curve(df['result'], df['sr_value'])
    j_scores = np.abs(tpr - fpr)
    j_ordered = sorted(zip(j_scores, thresholds))
    threshold = j_ordered[-1][1]

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



