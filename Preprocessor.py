
# This is Preprocessor Version 2
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def divide_train_validation_text(raws, train_offset, val_offset):
    return raws[:train_offset], raws[train_offset:val_offset], raws[val_offset:]


def show_class_distribution(labels, info):
    print(f'Label Distribution for {info}:')
    unique, counts = np.unique(labels, return_counts=True)
    all_cnt = sum(counts)
    for i in range(len(unique)):
        print(f"{unique[i]}: {(counts[i]/all_cnt):.3f}%")


def preprocess_data(data, is_binary=True, is_supervised=True, train_portion = 0.38 , validation_portion=0.42):
    # select label classes
    labels_binary = data['NST_B_Label'].values
    labels_multi = data['NST_M_Label'].values
    label_encoder = LabelEncoder()
    labels_multi = label_encoder.fit_transform(labels_multi)

    # drop extra and label columns
    columns_to_drop = [
        'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'start', 'end',
        'startDate', 'endDate', 'startOffset', 'endOffset', 'IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label'
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    # apply preprocessing
    data.fillna(0, inplace=True)
    data['protocol'] = data['protocol'].astype(str)
    encoder = OneHotEncoder(sparse_output=False)
    protocols_encoded = encoder.fit_transform(data[['protocol']])
    data = pd.concat([data.drop('protocol', axis=1),
                      pd.DataFrame(protocols_encoded, columns=encoder.get_feature_names_out(['protocol']))],
                     axis=1)
    # 64 all features - 12(time and address)  - 4(label features) -1(protocol) + 5(protocol encoded) = 52 features

    if is_binary:
        train_offset = int(train_portion * len(data))
        val_offset = train_offset + int(validation_portion * len(data))
    else:
        train_offset = int(0.7 * len(data))
        val_offset = train_offset + int(0.1 * len(data))

    if not is_binary:
        x_train, x_val, x_test = divide_train_validation_text(data, train_offset, val_offset)
        y_train, y_val, y_test = divide_train_validation_text(labels_multi, train_offset, val_offset)
    elif is_supervised:
        x_train, x_val, y_train, y_val = train_test_split(data[:val_offset], labels_binary[:val_offset],
                                                          test_size=(1 - (train_offset / val_offset)),
                                                          random_state=42)
        x_test, y_test = data[val_offset:], labels_binary[val_offset:]
    else:
        x_train, x_val, x_test = divide_train_validation_text(data, train_offset, val_offset)
        y_train, y_val, y_test = divide_train_validation_text(labels_binary, train_offset, val_offset)

    # Scaling
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    show_class_distribution(y_train, 'train')
    show_class_distribution(y_val, 'validation')
    show_class_distribution(y_test, 'test')

    print(f'data shape: {data.shape}')
    print(f'x_train shape: {x_train.shape}')
    print(f'y_val shape: {y_val.shape}')
    print(f'y_test shape: {y_test.shape}')

    return x_train, x_val, x_test, y_train, y_val, y_test
