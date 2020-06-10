import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

train_windows = pd.read_csv('train-forecasts.csv', index_col=False)
test_windows = pd.read_csv('test-forecasts.csv', index_col=False)

train_data_without_attacks = train_windows.loc[train_windows['percentage'] == 1]

train_windows = train_windows[0:len(train_data_without_attacks)]
train_result = train_windows['result']

train_data = train_windows.drop(['result', 'percentage', 'slot', 'duration'], 1)
train_data_original = train_data
test_result = test_windows['result']
test_data_percentages = test_windows['percentage']
test_data_slots = test_windows['slot']
test_data_duration = test_windows['duration']
test_data = test_windows.drop(['result', 'percentage', 'slot', 'duration'], 1)
test_data_original = test_data

sc = StandardScaler()

result = pd.DataFrame()
result['accuracy'] = 0
result['precision'] = 0
result['recall'] = 0
result['f1'] = 0
result['fpr'] = 0
for i in range(10):
    print ('Iteration ' + str(i+1))
    train_data = sc.fit_transform(train_data_original)
    test_data = sc.transform(test_data_original)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(train_data.shape[1], 1)))
    model.add(tf.keras.layers.Conv1D(48, 7, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Conv1D(24, 7, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(90, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    model.fit(train_data, train_result, epochs=2000, validation_split=0.2, verbose=0, callbacks=[early_stop])
    y_pred = model.predict(test_data)
    y_pred = np.round(y_pred)

    y_pred = [x[1] for x in y_pred]
    tn, fp, fn, tp = confusion_matrix(df['result'], df['prediction']).ravel()
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / len(df)
    recall = tp / (tp + tn)
    f1 = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    result.loc[i] = [accuracy, precision, recall, f1, fpr]

print(result.mean())
