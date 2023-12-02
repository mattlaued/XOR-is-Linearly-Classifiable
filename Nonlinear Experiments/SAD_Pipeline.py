import pandas as pd
from sklearn.preprocessing import StandardScaler
from SupervisedAD_methods import *


import sys

sys.path.append("./mvtec")
from methods import get_metrics, evaluate_predictions


strategy = tf.distribute.MirroredStrategy()


def get_data(data_dir, train_file, test_file,
             normal_labels, seen_anom_label,
             anom_label):
    df = pd.read_csv(f"{data_dir}/{train_file}", header=None, delimiter=' ').dropna(axis=1)
    df_full = df.to_numpy()
    print("Value Counts for Each Class")
    print(np.unique(df_full[:,-1], return_counts=True))
    df_test = pd.read_csv(f"{data_dir}/{test_file}", header=None, delimiter=' ').dropna(axis=1)
    df_full_test = df_test.to_numpy()
    print(np.unique(df_full_test[:,-1], return_counts=True))
    labels = df_full[:, -1]

    scaler = StandardScaler()

    x_normal_train = df_full[np.isin(labels, normal_labels)][:, :-1]
    x_seen_anom_train = df_full[labels == seen_anom_label][:, :-1]

    x_train = scaler.fit_transform(np.vstack((x_normal_train, x_seen_anom_train)))
    y_train = np.hstack(
        (np.zeros(len(x_normal_train)), np.ones(len(x_seen_anom_train)))
    )

    print("Training: Normal VS Seen Anomalies")
    print(x_normal_train.shape, x_seen_anom_train.shape)

    data_unseen_anom = df_full[~np.isin(labels, normal_labels + [seen_anom_label])]
    x_unseen_anom = data_unseen_anom[:, :-1]
    unseen_anom_labels = data_unseen_anom[:, -1]

    x_test = scaler.transform(np.vstack((df_full_test[:, :-1], x_unseen_anom)))
    y_test = np.hstack(
        (~np.isin(df_full_test[:, -1], normal_labels), np.ones(len(unseen_anom_labels)))
    )

    anom_ids = set(np.unique(df_full_test[:, -1]).astype(int))
    anom_ids -= set(normal_labels)
    anom_ids = sorted(list(anom_ids))
    anom_label_data = y_test * np.hstack(
        (df_full_test[:, -1], unseen_anom_labels)
    )

    print(anom_ids, anom_label, anom_label_data)

    return x_train, y_train, x_test, y_test, anom_ids, anom_label, anom_label_data


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
        callbacks=[]
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
                                                            anom_label_data=anom_label_data)

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