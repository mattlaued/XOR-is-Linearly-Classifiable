import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib
import csv

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss',
                                                  restore_best_weights=True)

lim = 2.25
grid = int(lim*100*2)
a = np.linspace(-lim, lim, grid)
xv, yv = np.meshgrid(a, a)
data_viz = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))

radius_sq = (data_viz ** 2).sum(axis=1)
radius2 = (np.abs(radius_sq - 4) < 1e-1).reshape(grid, grid)
radius1 = (np.abs(radius_sq - 1) < 5e-2).reshape(grid, grid)
radius05 = (np.abs(radius_sq - 0.25) < 3e-2).reshape(grid, grid)

# functions to generate data
# unit circle

def generate_data(total, ratio=0.5, shuffle=True, save=False, fake_r=2, seed=100):
    '''
    save: False, "train" or "test"
    '''
    
    assert abs(ratio) <= 1
    
    true_x = generate_x(int(total * ratio), r=1, seed=seed)
    true_data = np.hstack((true_x, np.ones(int(total * ratio)).reshape(-1, 1)))
    
    fake_x = generate_x(total - int(total * ratio), r=fake_r, seed=seed+42)
    fake_data = np.hstack((fake_x, np.zeros(total - int(total * ratio)).reshape(-1, 1)))
    
    x_all = np.vstack((true_data, fake_data))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(x_all)
        
    if save:
        with open(f"data/2Dcircle_{save}{total}.npy", "wb") as f:
            np.save(f, x_all)
    
    return x_all


def generate_angle(n, seed=100):
    
    np.random.seed(seed)
    
    return np.random.uniform(0, 360, size=(n,1))


def generate_x(n, r=1, seed=100):
    '''
    r is a number, use as radius
    else, if r is a list of 2 numbers, randomly generate numbers between those 2 numbers
    '''
    
    theta = generate_angle(n, seed=seed+1)
    
    if type(r) is list:
        np.random.seed(seed)
        r = np.random.uniform(r[0], np.sqrt(r[1]), size=(n,1))
    
    return np.hstack((r**2 * np.cos(theta), r**2 * np.sin(theta)))

# Visualization and Evaluation

def viz_boundary(data_viz, model, grid=grid, writer=False):
    '''writer: False or csv writer'''
    
    y_pred = model(data_viz).numpy()

    transparency = 0.7
    plt.pcolormesh(radius2, cmap="YlGn", alpha=transparency)
    plt.pcolormesh(radius1, cmap="Greys", alpha=transparency)
    plt.pcolormesh(radius05, cmap="RdPu", alpha=transparency)

    y_im = y_pred.reshape(grid, grid)
    norm = matplotlib.colors.Normalize(y_pred.min(), y_pred.max())
    plt.pcolormesh(y_im, cmap="hot", norm=norm, alpha=0.6)
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    plt.title("Histogram for Predictions on Visualization Grid")
    y_pred_pos = y_pred[radius1.reshape(-1) == 1]
    y_pred_neg = y_pred[radius1.reshape(-1) == 0]
    plt.hist(y_pred_neg, bins=10, label="Negative", alpha=0.5)
    plt.hist(y_pred_pos, bins=10, label="Positive", alpha=0.5)
    plt.legend()
    plt.show()
    
    if writer:
        writer.writerow(np.hstack([[model.name], y_pred.squeeze()]))
    
    
def viz_sigma(weight_callback):
    
    for k, v in weight_callback.weight_dict.items():
    
        plt.plot(v, label=k)
        plt.title("Sigma against Epoch")
        plt.ylabel("Sigma")
        plt.xlabel("Epoch number")
    
    plt.legend()
    plt.show()

    
def get_metrics(y_pred, y_test, model_name, plot=True, pos_label=0):

#     y_pred = model(x_test)
    sign = 1

    # change pos_label from 1 to 0 since positive is attack
    if pos_label == 0:
        sign = -1
    fpr, tpr, threshold = metrics.roc_curve(y_test, sign * y_pred, pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, threshold = metrics.precision_recall_curve(
        y_test, -y_pred, pos_label=0)
    pr_auc = metrics.auc(recall, precision)
    
    if plot:

        fig, axs = plt.subplots(nrows=1, ncols=2)

        axs[0].plot(fpr, tpr, alpha=0.5,
                 label=f'AUC = %0.2f' % roc_auc)
        axs[1].plot(recall, precision, alpha=0.5,
                 label=f'AUC = %0.2f' % pr_auc)

        axs[0].set_title(f'ROC Curve: {model_name}')
        axs[0].legend(loc = 'lower right')
        axs[0].plot([0, 1], [0, 1],'r--')
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_xlabel('False Positive Rate')

        axs[1].set_title(f'Precision-Recall Curve: {model_name}')
        axs[1].legend(loc = 'lower right')
        axs[1].axhline(1 - y_test.sum()/len(y_test), linestyle='--', color='r')
        axs[1].set_xlim([0, 1])
        axs[1].set_ylim([0, 1])
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall')

        plt.tight_layout()
        plt.show()
    
    return pr_auc

# Shallow Model Evaluation

def eval_plot(y_test, y_pred, model_name, plot=True):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred, pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, threshold = metrics.precision_recall_curve(
        y_test, y_pred, pos_label=0)
    pr_auc = metrics.auc(recall, precision)
    
    if plot:

        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(9, 4.5)

        axs[0].set_title('ROC Curve')
        axs[0].plot(fpr, tpr, 'purple', alpha=0.8,
                 label=f'{model_name} AUC = %0.2f' % roc_auc)
        axs[0].legend(loc = 'lower right')
        axs[0].plot([0, 1], [0, 1],'r--')
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_xlabel('False Positive Rate')

        axs[1].set_title('Precision-Recall Curve')
        axs[1].plot(recall, precision, 'purple', alpha=0.8,
                 label=f'{model_name} AUC = %0.2f' % pr_auc)
        axs[1].legend(loc = 'lower right')
        axs[1].axhline(y_test.sum()/len(y_test), linestyle='--', color='r')
        axs[1].set_xlim([0, 1])
        axs[1].set_ylim([0, 1])
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall')

        plt.tight_layout()
        plt.show()
    
    return pr_auc


# Bumped Perceptron
def not_that_deep(X, y, sigma=10., loss="mse", seed=0):
    
    # x = [1, x_1, x_2]

    X_ = np.vstack((X.shape[0] * [1], X.T)).T
    
    np.random.seed(seed)
    beta = np.random.randn(X_.shape[1])

    def xorloss(beta):
        y_pred = np.exp(-0.5 * np.square(X_.dot(beta) / sigma)) 

        # without smoothing
        # y_pred = X_.dot(beta) != 0
        
        if loss == "mse":
            return np.linalg.norm(y - y_pred) ** 2
        elif loss == "bce":
            return metrics.log_loss(y, y_pred)
        else:
            return loss(y, y_pred)

    res = minimize(xorloss, beta)

    return res.x

# Bump Activation

def bump_activation(x, sigma=0.5, type_of_bump='gaussian', number_of_margins:int=1):
    """
    bump activation
    :param x: input
    :param sigma: variance / width parameter
    :param type_of_bump: 'gaussian' or 'tanh'
    :param number_of_margins: number of separate margins, int
    :return:
    """

    if number_of_margins == 1:
        if type_of_bump == 'gaussian':
            return tf.math.exp(-0.5 * tf.math.square(x) / tf.math.square(sigma))
        return tf.math.tanh(tf.math.square(sigma/x))
    s = sum([
        bump_activation(x-i*10, sigma, number_of_margins=1) for i in range(number_of_margins)
    ])
    # normalise so max is 1
    middle = number_of_margins // 2
    normalise = sum([
        bump_activation((middle-i)*10, sigma, number_of_margins=1) for i in range(number_of_margins)
    ])
    return s / normalise


# Deep Model Training

class Bump(tf.keras.layers.Layer):

    def __init__(self, sigma=0.5, trainable=True, name="bump", **kwargs):
        super(Bump, self).__init__(**kwargs)
        self.supports_masking = True
        self.sigma = sigma
        self.trainable = trainable
        self._name = name

        
    def build(self, input_shape):
        self.sigma_factor = K.variable(self.sigma,
                                      dtype=K.floatx(),
                                      name='sigma_factor')
        if self.trainable:
            self._trainable_weights.append(self.sigma_factor)

        super(Bump, self).build(input_shape)

    def call(self, inputs, mask=None):
        return bump_activation(inputs, self.sigma_factor)

    def get_config(self):
        config = {'sigma': self.get_weights()[0] if self.trainable else self.sigma,
                  'trainable': self.trainable}
        base_config = super(Bump, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    

class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma, initializer, dim=1, beta_param=False, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
        self.initializer = initializer
        self.beta_param = beta_param

        
        # dim is 1D (tabular) or 2D (sequence)
        self.dim = dim
        
    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[-1]), self.units),
                                  initializer=self.initializer,
                                  trainable=True)
        if self.beta_param:
            self.beta = self.add_weight(name='beta',
                                  shape=(1,),
                                  initializer=self.initializer,
                                  trainable=True)
        else:
            self.beta = 1.
        super(RBFLayer, self).build(input_shape)
        
    def call(self, inputs):
        if self.dim == 1:
            diff = K.expand_dims(inputs) - self.mu
            l2 = K.sum(K.pow(diff/self.beta,2), axis=1)
        else:
            x = inputs[:, :, :, None]
            mu = self.mu[None, None, :, :]
            diff = tf.subtract(x, mu)
            l2 = K.sum(K.pow(diff/self.beta,2), axis=-2)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        if self.dim == 1:
            return (input_shape[0], self.units)
        return input_shape[:-1] + (self.units,)

    
class GetWeights(tf.keras.callbacks.Callback):
    
    def __init__(self, layer_names=["bump"]):
        super(GetWeights, self).__init__()
        self.weight_dict = {}
        self.layer_names = layer_names
        for layer in layer_names:
            self.weight_dict[layer] = []

    def on_epoch_end(self, epoch, logs=None):        
        if len(self.layer_names) > 0:            
            for layer in self.layer_names:                             
                self.weight_dict[layer].append(self.model.get_layer(layer).get_weights()[0])
    
    
# Build Models
def build_layer(activation, input_layer, sigma=0.5, train=False, layer_number=1, seed=0, neurons=5, batchnorm=False):
    
    initialiser = tf.keras.initializers.GlorotUniform(seed=seed)
    
    if activation == "r":
        layer = RBFLayer(neurons, gamma=1.0, initializer=initialiser)(input_layer)
        
        if batchnorm:
            layer = tf.keras.layers.BatchNormalization()(layer)
            
    else:
        hidden = tf.keras.layers.Dense(neurons,
                      kernel_initializer=initialiser)(input_layer)
        
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


def create_model(separation, activation, hidden_layers, input_dim=2,
                 sigma=0.5, train=False, loss='binary_crossentropy',
                 seed=0, name_suffix=""):
    
    sep = {"RBF": "r", "ES": "b", "HS": "s"}
    
    tf.keras.utils.set_random_seed(seed)

    input_layer = tf.keras.Input(shape=(input_dim,))
    hidden1 = build_layer(activation, input_layer, sigma=sigma, train=train, layer_number=1, seed=seed+42)
    hidden2 = build_layer(activation, hidden1, sigma=sigma, train=train, layer_number=2, seed=seed+123)
    
    if hidden_layers == 2:
    
        out = build_layer(sep[separation], hidden2, sigma=sigma, layer_number="last", seed=seed+2023, neurons=1)
    
    elif hidden_layers == 3:
        
        hidden3 = build_layer(activation, hidden2, sigma=sigma, train=train, layer_number=3, seed=seed+1234)   
        out = build_layer(sep[separation], hidden3, sigma=sigma, layer_number="last", seed=seed+2023, neurons=1)

    model = tf.keras.Model(inputs=input_layer, outputs=out, name=f'{separation}{hidden_layers}{activation}{name_suffix}')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss=loss)
    
    return model


# Train and Evaluate Models
def train_eval(model, X, y, x_test, y_test, train=False, hidden_layers=2, verbose=0, shuffle=False, plot=False,
               val_split=0.1, callbacks=[early_stopping], seed=0, diff=True, writer=False):
    
    # Train the model
    tf.keras.utils.set_random_seed(seed)
    cbs = callbacks.copy()
    if train:
        get_weights = GetWeights(layer_names=[f"bump{i}" for i in range(1, hidden_layers+1)])
        cbs.append(get_weights)
        model.fit(X, y, epochs=1000, verbose=verbose, shuffle=shuffle,
                  validation_split=val_split, callbacks=cbs)
        viz_sigma(get_weights)
        
    else:
        model.fit(X, y, epochs=1000, verbose=verbose, shuffle=shuffle,
                  validation_split=val_split, callbacks=cbs)

    # Evaluation
    viz_boundary(data_viz, model, grid=grid, writer=writer)
    aupr_train = get_metrics(X, y, model, plot=plot)
    aupr_test = get_metrics(x_test, y_test, model, plot=plot)

    plt.title("Histogram for Predictions on Test Data")
    y_pred = model(x_test).numpy()
    y_pred_pos = y_pred[y_test == 1]
    y_pred_neg = y_pred[y_test == 0]
    plt.hist(y_pred_neg, bins=10, label="Negative", alpha=0.5)
    plt.hist(y_pred_pos, bins=10, label="Positive", alpha=0.5)
    plt.legend()
    plt.show()
    
    if diff:
        y_pos = np.mean(y_pred_pos)
        y_neg = np.mean(y_pred_neg)
        diff_mean = y_pos - y_neg
        print(f"Average Difference between Positive and Negative Class: {diff_mean}")
        
        return aupr_train, aupr_test, diff_mean
    
    return aupr_train, aupr_test
