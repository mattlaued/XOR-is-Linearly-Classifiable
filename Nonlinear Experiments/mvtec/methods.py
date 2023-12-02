import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


def get_dataset(categories, data_dir, model_type, agg_method, k=1):
    # The first k categories are considered good, all other are defect
    label_list = []
    img_emb_list = []
#     file_names = []
    for i, name in enumerate(categories):
        # label normal as 0, anomalies as 1
        if i < k:
            label = 0
        else:
            label = 1

        # Get all files that match pattern 
        for file in glob.glob(f'{data_dir}/embeddings/{model_type}-{agg_method}-{name}-*.npy'):
#             file_names.append(file)
            label_list.append(label)
            arr = np.load(file)
            img_emb_list.append(arr)
            
    # get CLS token
    X = np.array(img_emb_list)
    y = np.array(label_list)
#     print(file_names)

    return X, y


def bump_activation(x, sigma=0.5):
    return torch.exp(-0.5 * torch.square(x) / torch.square(sigma))


class Bump(nn.Module):
    def __init__(self, sigma=0.5, trainable=False):
        super(Bump, self).__init__()
        self.sigma = sigma
        self.sigma_factor = nn.Parameter(
            torch.tensor(self.sigma, dtype=torch.float32), requires_grad=trainable)

    def forward(self, inputs):
        return bump_activation(inputs, self.sigma_factor)


class RBFLayer(nn.Module):
    def __init__(self, units, gamma, initializer=nn.init.xavier_normal_, dim=1, seed=2023):
        super(RBFLayer, self).__init__()
        self.units = units
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.dim = dim
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if dim == 1:
            self.mu = nn.Parameter(initializer((units,)), requires_grad=False)
        else:
            self.mu = nn.Parameter(initializer(units), requires_grad=False)

    def forward(self, inputs):
        if self.dim == 1:
            inputs_expanded = torch.unsqueeze(inputs, 1)
            diff = inputs_expanded - self.mu
            l2 = torch.sum(torch.pow(diff, 2), dim=2)
        elif self.dim == 2:
            x = inputs[:, :, None]
            mu = self.mu[None, None, :]
            diff = x - mu
            l2 = torch.sum(torch.pow(diff, 2), dim=-1)
        elif self.dim == 4:
            # Reshape inputs to (batch_size, channels, height * width)
            inputs_reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)

            # Calculate L2 distance between inputs and centroids
            diff = inputs_reshaped[:, None, :, :] - self.centroids[None, :, :, :]
            l2 = torch.sum(torch.pow(diff, 2), dim=3)
            l2 = torch.sum(l2, dim=2)
        res = torch.exp(-1 * self.gamma * l2)
        return res


def build_layer(activation, input_layer, layer_type="linear", sigma=0.5, train=False,
                neurons=5, batchnorm=False, regularizer=None, initializer=nn.init.xavier_uniform_):
    """

    :param activation: r for RBF, b for Bump, s for sigmoid and l for leaky ReLU
    :param input_layer: input layer (tensor)
    :param layer_type: linear or conv
    :param sigma: for bump activation (if applicable)
    :param train: trainable sigma for bump (if applicable)
    :param neurons: unit shape for output
    :param batchnorm: bool of whether to apply batchnorm
    :param regularizer: None or regulariser
    :param initializer: initialiser for weights
    :return: output tensor of the layer
    """

    input_dim = input_layer.size(1)

    if activation == "r":
        layer = RBFLayer(neurons, gamma=1.0, initializer=initializer)(input_layer)

        if batchnorm:
            if layer_type == "linear":
                layer = nn.BatchNorm1d(neurons)(layer)
            else:
                layer = nn.BatchNorm2d(neurons)(layer)

    else:
        if layer_type == "linear":
            hidden = nn.Linear(input_dim, neurons)
            initializer(hidden.weight)
            if regularizer is not None:
                hidden.weight_regularizer = regularizer
            input_layer = input_layer.view(-1, input_dim)
            hidden_out = hidden(input_layer)
        elif layer_type == "conv":
            # Create a 2D convolutional layer
            conv2d = nn.Conv2d(input_dim, neurons, kernel_size=3, stride=1, padding=1)
            initializer(conv2d.weight)
            if batchnorm:
                conv2d = nn.Sequential(conv2d, nn.BatchNorm2d(neurons))
            hidden_out = conv2d(input_layer)
        else:
            raise Exception("please input valid layer type")

        if batchnorm:
            hidden_out = nn.BatchNorm1d(neurons)(hidden_out)

        if activation == "b":
            layer = Bump(sigma=sigma, trainable=train)(hidden_out)
        elif activation == "s":
            layer = torch.sigmoid(hidden_out)
        else:
            layer = F.leaky_relu(hidden_out, negative_slope=0.01)

    return layer


def replace_layers(model, old, new):
    """
    replace layers (eg. in pre-trained model)
    :param model: model obj
    :param old: old layer (eg. nn.ReLU)
    :param new: new layer
    :return: None (model obj is edited)
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            print(module)
            ## simple module
            setattr(model, n, new)


def not_that_deep(X, y, sigma=10., activation='bump',
                  loss="mse", seed: int = 0):
    """

    :param X: data
    :param y: labels
    :param sigma: scaling factor
    :param activation: last layer activation
    :param loss: loss function (mse/bce string or function)
    :param seed: random seed
    :return: weights of shallow model
    """

    # x = [1, x_1, x_2, ...]
    X_ = np.vstack((X.shape[0] * [1], X.T)).T

    np.random.seed(seed)
    beta = np.random.randn(X_.shape[1])
    if activation == "rbf":
        beta = beta[:-1]

    def local_loss(beta):
        if activation == "bump":
            y_pred = np.exp(-0.5 * np.square(X_.dot(beta) / sigma))
        elif activation == "rbf":
            y_pred = np.exp(np.linalg.norm(X - beta, axis=1) / sigma)
        else:
            y_pred = activation(X)
        # without smoothing
        # y_pred = X_.dot(beta) != 0

        if loss == "mse":
            return np.linalg.norm(y - y_pred) ** 2
        elif loss == "bce":
            return metrics.log_loss(y, y_pred)
        else:
            return loss(y, y_pred)

    res = minimize(local_loss, beta)

    return res.x


def prober(x, beta, probe="bump", sigma=10.):
    """
    @param x: 2D array of examples (num, dim)
    @param beta: 1D array of learnt parameter (dim,)
    @param probe: string for 'bump' (hyperplane) or 'rbf' (point)
    """
    
    if probe == "bump":
        # add 1 for bias
        x_withbias = np.vstack((x.shape[0] * [1], x.T)).T
        return np.exp(-0.5 * np.square(x_withbias.dot(beta) / sigma))
    
    if probe == "rbf":
        return np.exp(np.linalg.norm(x - beta, axis=1) / sigma)
    
    raise ValueError("Probe not valid!")

    
def evaluate_probe(X_train, y_train, x_test, y_test, anom_ids, anom_label, anom_label_data,
                   probe="bump", loss="bce", sigma=10., seed=42):
    """
    @param X_train: 2D array of training examples (num, dim)
    @param y_train: 1D array of training labels. 0: normal, 1: anomaly.
    @param X_test: 2D array of test examples (num, dim)
    @param y_test: 1D array of test labels. 0: normal, 1: anomaly.
    @param probe: string for 'bump' (hyperplane) or 'rbf' (point)
    @param loss: string for 'bce' (binary cross-entropy) or 'mse'
    @param probe: string for 'bump' (hyperplane) or 'rbf' (point)
    """
    
    if probe == "bump":
        model_name = "Hyperplane Probe"
    elif probe == "rbf":
        model_name = "Point Probe"
    else:
        raise ValueError("Probe not valid! Should be string for 'bump' (hyperplane) or 'rbf' (point).")
    
    beta = not_that_deep(X_train, 1 - y_train, sigma=sigma, activation=probe, loss=loss, seed=seed)
    
    print("Train")
    y_pred = prober(X_train, beta, probe=probe, sigma=sigma)
    aupr_train = evaluate_predictions(y_pred, 1-y_train,
                                      model_name=model_name, plot=True, pos_label=0)

    print("Test")
    y_pred = prober(x_test, beta, probe=probe, sigma=sigma)
    aupr_test, aupr_indiv_anoms = evaluate_predictions(y_pred, 1-y_test, model_name=model_name, plot=True, diff=False,
                 indiv=anom_ids, anom_label=anom_label, anom_label_data=anom_label_data, pos_label=0)
    
    return beta, aupr_test, aupr_indiv_anoms


def get_metrics(y_pred, y_test, model_name, plot=True, pos_label=0, save=False):
    """

    :param y_pred: label predictions
    :param y_test: label ground truth
    :param model_name: name of model
    :param plot: whether to plot ROC and PR curves
    :param pos_label: attack label
    :param save: False to not save AUROC/AUPR plots, else path string to save
    :return: AUPR (Area under Precision-Recall Curve)
    """
    sign = 1

    # change pos_label from 1 to 0 since positive is attack
    if pos_label == 0:
        sign = -1
    fpr, tpr, threshold = metrics.roc_curve(y_test, sign * y_pred, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, threshold = metrics.precision_recall_curve(
        y_test, sign * y_pred, pos_label=pos_label)
    pr_auc = metrics.auc(recall, precision)

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2)

        axs[0].plot(fpr, tpr, alpha=0.5,
                    label=f'AUROC = %0.5f' % roc_auc)
        axs[1].plot(recall, precision, alpha=0.5,
                    label=f'AUPR = %0.5f' % pr_auc)

        axs[0].set_title(f'ROC Curve: {model_name}')
        axs[0].legend(loc='lower right')
        axs[0].plot([0, 1], [0, 1], 'r--')
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_xlabel('False Positive Rate')

        axs[1].set_title(f'Precision-Recall Curve: {model_name}')
        axs[1].legend(loc='lower right')
        axs[1].axhline(1 - y_test.sum() / len(y_test), linestyle='--', color='r')
        axs[1].set_xlim([0, 1])
        axs[1].set_ylim([0, 1])
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall')

        plt.tight_layout()
        if save:
            plt.savefig(save + "/Overall_AUPR_AUROC.png")
        plt.show()


    return pr_auc


def evaluate_predictions(y_pred, y_test, model_name="Model",
             plot=False, diff=False,
             indiv=False, anom_label=None, anom_label_data=None,
             pos_label=0, save=False):
    """

    :param y_pred: array of label predictions
    :param y_test: array of label ground truth
    :param model_name: name of model
    :param plot: bool whether to plot ROC and PR curves
    :param diff: bool whether to calculate avg diff btw pos and neg predictions
    :param indiv: list of anomaly ID describing individual anomaly types
    :param anom_label: list/dict of anomaly names (str) describing individual anomaly types
    :param anom_label_data: arr of anomaly ID of each sample in y_test
    :param pos_label: attack label
    :param save: False to not save AUROC/AUPR plots, else path string to save
    :return: AUPR (and indiv or diff results, if applicable)
    """

    # Evaluation
    aupr_test = get_metrics(y_pred, y_test, model_name, plot=plot, pos_label=pos_label, save=save)

    plt.title("Histogram for Predictions on Test Data")
    y_pred_normal = y_pred[y_test == (1 - pos_label)].squeeze()
    y_pred_anomalies = y_pred[y_test == pos_label].squeeze()
    plt.hist(y_pred_anomalies, bins=20, label="Anomalies", alpha=0.5)
    plt.hist(y_pred_normal, bins=20, label="Normal", alpha=0.5)
    plt.legend()
    if save:
        plt.savefig(f'{save}_{model_name}_test_pred.png')
    if plot:
        plt.show()
    else:
        plt.close()

    if indiv:
        # get indiv auprs for different attacks
        y_normal = len(y_pred_normal)

        aupr_attacks = []
        for i, anom in enumerate(indiv):
            y_anom = y_pred[anom_label_data == anom].squeeze()
            #             print(y_pred_pos.shape)
            #             print(y_att.shape)
            #             print(y_normal)
            aupr_attack = get_metrics(
                    np.hstack((y_pred_normal, y_anom)),
                    np.hstack((np.ones(y_normal), np.zeros(len(y_anom)))),
                    model_name, plot=plot)
            aupr_attacks.append(aupr_attack)
        fig, ax = plt.subplots(nrows=1, ncols=len(indiv), figsize=(3.5 * len(indiv), 7))
        for i, anom in enumerate(indiv):
            y_anom = y_pred[anom_label_data == anom].squeeze()
            #             print(y_pred_pos.shape)
            #             print(y_att.shape)
            #             print(y_normal)

            ax[i].set_title(f"Anomaly Predictions")
            ax[i].hist(y_anom, bins=20, label=f"Anomaly {anom}: {anom_label[anom]}", alpha=0.5)
            ax[i].hist(y_pred_normal, bins=20, label="Normal", alpha=0.5)
            ax[i].text(0.0, 0.9, f'AUPR: {aupr_attacks[i]:.5}', style='italic', horizontalalignment='left',
                verticalalignment='center', transform=ax[i].transAxes)
            ax[i].legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"{save}_FineGrained_AUPR_AUROC.png")
        if plot:
            plt.show()
        else:
            plt.close()

        try:
            display(pd.DataFrame(data={model_name: aupr_attacks}))
        except:
            print(pd.DataFrame(data={model_name: aupr_attacks}))


        return aupr_test, aupr_attacks

    if diff:
        y_pos = np.mean(y_pred_normal)
        y_neg = np.mean(y_pred_anomalies)
        diff_mean = y_pos - y_neg
        print(f"Average Difference between Positive and Negative Class: {diff_mean}")

        return aupr_test, diff_mean

    return aupr_test
