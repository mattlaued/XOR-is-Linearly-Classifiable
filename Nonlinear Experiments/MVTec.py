import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc

from SupervisedAD_methods import *

import sys

sys.path.append("./mvtec/")
from methods import get_dataset, get_metrics, evaluate_predictions

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trial", action="store_true")
parser.add_argument("--deep", action="store_true")
args = parser.parse_args()

epochs = 500
if args.deep:
    NEURONS = [5, 5]
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    epochs = 1000
    sigma = 5.
else:
    NEURONS = []
    lr = 3e-4
    sigma = 3.

if args.trial:
    repeats = 2
    epochs = 2
    verbose = 1
    # DATASETS = ['capsule', 'hazelnut', 'pill']
    DATASETS = ['cable', 'carpet']
else:
    repeats = 3
    verbose = 0
    # DATASETS = ['tile']
    DATASETS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


MODELS = {  # model of similar size (1.22 GB)
    'dinov2': ('facebook/dinov2-large', ['cls'], True)
}
# MODELS = {  # model of similar size (1.22 GB)
#     'vit': ('google/vit-large-patch16-224-in21k', ['mean'], True)
# }
# MODELS = {# model of similar size (1.22 GB)
#     'dinov2': ('facebook/dinov2-large', ['cls', 'mean'], True),
#     'clip': ('openai/clip-vit-large-patch14', ['cls'], True),
#     'mae': ('facebook/vit-mae-large', ['mean'], False),
#     'vit': ('google/vit-large-patch16-224-in21k', ['cls', 'mean'], True)
#     }

strategy = tf.distribute.MirroredStrategy()


# dataset = "wood"
# data_dir = f"../Anomaly Detection/mvtec_anomaly_detection/{dataset}"
# model_type = "dinov2"
# agg_method = "cls"


# Build Models
def build_layer(activation, input_layer, sigma=0.5, train=False, layer_number=1,
                seed=0, neurons=5, batchnorm=False, regulariser=None):
    initialiser = tf.keras.initializers.GlorotUniform(seed=seed)

    if activation == "r":
        layer = RBFLayer(neurons, gamma=1.0, initializer=initialiser)(input_layer)

        if batchnorm:
            layer = tf.keras.layers.BatchNormalization()(layer)

    else:
        hidden = tf.keras.layers.Dense(neurons,
                                       kernel_initializer=initialiser, kernel_regularizer=regulariser)(input_layer)

        if batchnorm:
            hidden = tf.keras.layers.BatchNormalization()(hidden)

        if activation == "b":
            layer = Bump(sigma=sigma, trainable=train,
                         name=f"bump{layer_number}")(hidden)
        elif activation == "s":
            layer = tf.math.sigmoid(hidden)
        else:
            layer = tf.nn.leaky_relu(hidden, alpha=0.01)

    return layer


def create_model(separation, activation, hidden_layers, num_inputs,
                 hidden_neurons=[40, 20, 10, 5], dropout=[0.0, 0.0, 0.0, 0.0], lr=0.001,
                 regularisation=[None, None, None, None],
                 sigma=0.5, train=False, loss='binary_crossentropy', batchnorm=False,
                 seed=0, name_suffix=""):
    sep = {"RBF": "r", "ES": "b", "HS": "s"}

    tf.keras.utils.set_random_seed(seed)

    input_layer = tf.keras.Input(shape=(num_inputs,))

    if type(hidden_neurons) is list:

        if len(hidden_neurons) == 0:
            out = build_layer(sep[separation], input_layer, sigma=sigma, layer_number="last", seed=seed + 2023,
                              neurons=1)

        else:

            hidden_layers = len(hidden_neurons)
            hidden = input_layer

            for i, n in enumerate(hidden_neurons):
                hidden = build_layer(activation, hidden, sigma=sigma, train=train,
                                     layer_number=1 + i, seed=seed + 42 * i, neurons=n,
                                     batchnorm=batchnorm, regulariser=regularisation[i])
                if dropout[i] > 0.:
                    hidden = tf.keras.layers.Dropout(dropout[i])(hidden)

            out = build_layer(sep[separation], hidden, sigma=sigma, layer_number="last", seed=seed + 2023, neurons=1)

    else:
        hidden1 = build_layer(activation, input_layer, sigma=sigma, train=train, layer_number=1, seed=seed + 42)
        hidden2 = build_layer(activation, hidden1, sigma=sigma, train=train, layer_number=2, seed=seed + 123)

        if hidden_layers == 2:

            out = build_layer(sep[separation], hidden2, sigma=sigma, layer_number="last", seed=seed + 2023, neurons=1)

        elif hidden_layers == 3:

            hidden3 = build_layer(activation, hidden2, sigma=sigma, train=train, layer_number=3, seed=seed + 1234)
            out = build_layer(sep[separation], hidden3, sigma=sigma, layer_number="last", seed=seed + 2023, neurons=1)

    model = tf.keras.Model(inputs=input_layer, outputs=out,
                           name=f'{separation}{hidden_layers}{activation}{name_suffix}')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=loss)

    return model


# Train and Evaluate Models

def train_eval(model, X, y, x_test, y_test, epochs=1000, train=False, hidden_layers=2,
               verbose=0, shuffle=False, plot=False,
               val_split=0.1, callbacks=[early_stopping], seed=0, diff=False, save=False,
               indiv=None, anom_label=None, anom_label_data=None, pos_label=0):
    """

    :param model:
    :param X:
    :param y: training label -- 0 for anom, 1 for normal
    :param x_test:
    :param y_test:
    :param epochs:
    :param train:
    :param hidden_layers:
    :param verbose:
    :param shuffle:
    :param plot:
    :param val_split:
    :param callbacks:
    :param seed:
    :param diff:
    :param save:
    :param indiv:
    :param anom_label:
    :param anom_label_data:
    :param pos_label:
    :return:
    """
    # Train the model
    tf.keras.utils.set_random_seed(seed)
    cbs = callbacks.copy()
    if train:
        # learnable sigma
        get_weights = GetWeights(layer_names=[f"bump{i}" for i in range(1, hidden_layers + 1)])
        cbs.append(get_weights)
        model.fit(X, y, epochs=epochs, verbose=verbose, shuffle=shuffle,
                  validation_split=val_split, callbacks=cbs)
        viz_sigma(get_weights)

    else:
        model.fit(X, y, epochs=epochs, verbose=verbose, shuffle=shuffle,
                  validation_split=val_split, callbacks=cbs)

    # Evaluation
    #     viz_boundary(data_viz, model, grid=grid, writer=writer)

    y_train = model.predict(X)
    aupr_train = get_metrics(y_train, y, model.name, plot=plot, pos_label=pos_label, save=save)

    y_pred = model.predict(x_test)
    aupr_test = evaluate_predictions(y_pred, y_test, model_name="Model",
                                     plot=False, diff=False,
                                     indiv=indiv, anom_label=anom_label, anom_label_data=anom_label_data,
                                     pos_label=0, save=save)

    if indiv or diff:
        return aupr_train, aupr_test[0], aupr_test[1]

    return aupr_train, aupr_test


def test_model(
        X_train, y_train, X_test, y_test, num_inputs,
        anom_ids=None, anom_label=None, anom_label_data=None,
        separation="ES",
        bumped="b",
        sigma=3.,
        train=False,  # train sigma. if NA, then False
        neurons=[],
        verbose=1,  # can change this to 0 to suppress verbosity during training
        plot=False,
        shuffle=False,
        val_split=0.0,
        repeats=3,
        epochs=500,
        batchnorm=True,
        lr=3e-4,
        # lr = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.005,
        #     decay_steps=100000,
        #     decay_rate=0.96,
        #     staircase=True)
        # early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss',
        #                                                   restore_best_weights=True)
        # callbacks = [early_stopping]
        callbacks=[],
        save=False
):
    hidden_layers = len(neurons)
    dropout = [0.0 for n in neurons]

    auprs_train = []
    auprs_test = []
    aupr_attacks = dict()

    models_hs = []

    # Train and Evaluate the Model
    for i in range(repeats):

        with strategy.scope():
            # Create the model
            tf.keras.utils.set_random_seed(i)

            model = create_model(separation, activation=bumped, hidden_layers=hidden_layers, num_inputs=num_inputs,
                                 hidden_neurons=neurons, batchnorm=batchnorm, dropout=dropout,
                                 sigma=sigma, train=train, loss='binary_crossentropy', lr=lr,
                                 seed=i)

            if i == 0:
                model.summary()

            # Train the model
            aupr_train, aupr_test, aupr_attack = train_eval(model, X_train, 1 - y_train, X_test, 1 - y_test,
                                                            epochs=epochs,
                                                            train=train, verbose=verbose, shuffle=shuffle, plot=plot,
                                                            val_split=val_split, callbacks=callbacks, seed=i,
                                                            indiv=anom_ids, anom_label=anom_label,
                                                            anom_label_data=anom_label_data, save=save)

        models_hs.append(model)

        print(f"AUPR Train Run {i + 1}: {aupr_train}")
        print(f"AUPR Test Run {i + 1}: {aupr_test}")
        print(f"AUPR Indiv Test Run {i + 1}: {aupr_attack}")
        auprs_train.append(aupr_train)
        auprs_test.append(aupr_test)
        aupr_attacks[i] = aupr_attack

    print(f"AUPR (Train): {np.mean(auprs_train)}+-{np.std(auprs_train)}")
    print(f"AUPR (Test): {np.mean(auprs_test)}+-{np.std(auprs_test)}")

    # last row is overall aupr
    results_df = pd.DataFrame(data=aupr_attacks, index=anom_ids)
    results_df.loc[anom_ids[-1] + 1] = auprs_test
    results_df['AUPR Mean'] = results_df.mean(axis=1)
    results_df['AUPR Std'] = results_df.iloc[:, :-1].std(axis=1)


    print(results_df)
    # print(f"Average Distance between Means: {np.mean(diff_means)}+-{np.std(diff_means)}")
    return results_df


def testing(data_dir, model_type, agg_method,
            sigma=3.,
            train=False,  # train sigma. if NA, then False
            neurons=[],
            verbose=1,  # can change this to 0 to suppress verbosity during training
            plot=False,
            shuffle=False,
            val_split=0.0,
            repeats=3,
            epochs=500,
            batchnorm=True,
            lr=3e-4
            ):
    # pull files for data
    unique_files = dict()
    for file in glob.glob(f'{data_dir}/embeddings/{model_type}-{agg_method}-*'):
        file_type = '-'.join(file.split('-')[2:4])
        if file_type in unique_files:
            unique_files[file_type] += 1
        else:
            unique_files[file_type] = 1

    # get datasets
    data_classes = sorted(list(unique_files.keys()))
    # print(data_classes)
    train_dirs = [data_classes[-1], data_classes[0]]
    test_dirs = data_classes[1:-1]
    test_dirs.remove("test-good")
    anom_ids = list(np.arange(len(test_dirs)) + 1)
    test_dirs = ['test-good'] + test_dirs
    print("train:", train_dirs)
    print("test:", test_dirs)

    X_train, y_train = get_dataset(train_dirs, data_dir=data_dir, model_type=model_type, agg_method=agg_method)

    X_test, y_test = get_dataset(test_dirs, data_dir=data_dir, model_type=model_type, agg_method=agg_method)

    # label anomaly info
    anom_label = test_dirs.copy()
    anom_label_data = np.hstack(
        [np.zeros(unique_files[test_dirs[0]])] + [np.ones(unique_files[test_dirs[i]]) * i for i in anom_ids])

    num_inputs = X_train.shape[-1]

    results = []

    for sep, activation in zip(
            ["ES", "HS", "RBF"], ["b", "", "r"]
    ):
        print(sep)

        results_df = test_model(
            X_train, y_train, X_test, y_test, num_inputs,
            anom_ids=anom_ids, anom_label=anom_label, anom_label_data=anom_label_data,
            separation=sep,
            bumped=activation,
            sigma=sigma,
            train=train,  # train sigma. if NA, then False
            neurons=neurons,
            verbose=verbose,  # can change this to 0 to suppress verbosity during training
            plot=plot,
            shuffle=shuffle,
            val_split=val_split,
            repeats=repeats,
            epochs=epochs,
            batchnorm=batchnorm,
            lr=lr,
            # lr = tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=0.005,
            #     decay_steps=100000,
            #     decay_rate=0.96,
            #     staircase=True)
            # early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss',
            #                                                   restore_best_weights=True)
            # callbacks = [early_stopping]
            callbacks=[],
            save=f"./results/{data_dir.split('/')[-1]}_{model_type}_{agg_method}_{sep}{activation}_neurons{'h'.join(str(x) for x in neurons)}"
        )

        results.append(results_df)

    result_comparison = pd.concat(results, axis=1)

    print()
    print("************************************************************************************")
    print(data_dir.split("/")[-1], model_type, agg_method)
    print(result_comparison)
    print("************************************************************************************")
    print()
    result_comparison.to_csv(
        f"results/{data_dir.split('/')[-1]}_{model_type}_{agg_method}_neurons{'h'.join(str(x) for x in neurons)}.csv",
        index=False)


for dataset in DATASETS:
    data_dir = f"../mvtec_anomaly_detection/{dataset}"
    for model_type, (url, agg, cls) in MODELS.items():
        for agg_method in agg:
            # # pull files for data
            # unique_files = dict()
            # for file in glob.glob(f'{data_dir}/embeddings/{model_type}-{agg_method}-*'):
            #     file_type = '-'.join(file.split('-')[2:4])
            #     if file_type in unique_files:
            #         unique_files[file_type] += 1
            #     else:
            #         unique_files[file_type] = 1
            #
            # # get datasets
            # data_classes = sorted(list(unique_files.keys()))
            # if len(data_classes) == 0:
            #     print(f'{data_dir}/embeddings/{model_type}-{agg_method}')
            print()
            print("###############################################################################")
            print(data_dir.split('/')[-1], model_type, agg_method)
            testing(data_dir, model_type, agg_method,
                    sigma=sigma,
                    train=False,  # train sigma. if NA, then False
                    neurons=NEURONS,
                    verbose=verbose,  # can change this to 0 to suppress verbosity during training
                    plot=False,
                    shuffle=False,
                    val_split=0.0,
                    repeats=repeats,
                    epochs=epochs,
                    batchnorm=True,
                    lr=lr)
