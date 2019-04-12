# Use scikit-learn to grid search the learning rate and momentum
import numpy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras import  layers, models, regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from earthquake.read_file import get_features
from keras.constraints import maxnorm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize(df):
    for c in (df.columns.values.tolist()):
        mean = df[c].mean()
        df[c] = df[c] - mean
        std = df[c].std()
        df[c] = df[c] / std
        return df

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', learn_rate=0.01, momentum=0, activation='relu' ,init_mode='uniform',
                dropout_rate = 0.1, dropout_rate2=0.1, weight_constraint = 0, neurons=150, neurons2=150, number_of_hd_layer=1):
    model = models.Sequential()
    if number_of_hd_layer >= 1:
        model.add(layers.Dense(neurons, input_dim=1121, activation=activation, kernel_constraint=maxnorm(weight_constraint),
                               kernel_regularizer=regularizers.l2(0.001), kernel_initializer=init_mode))
        model.add(layers.Dropout(dropout_rate))
    if number_of_hd_layer >= 2:
        model.add(
            layers.Dense(neurons2, input_dim=277, activation=activation, kernel_constraint=maxnorm(weight_constraint),
                         kernel_regularizer=regularizers.l2(0.001), kernel_initializer=init_mode))
        model.add(layers.Dropout(dropout_rate2))
    model.add(layers.Dense(1, activation="relu"))
    # Compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

X_train, y_train, X_test = get_features()
X_test_seg_id = X_test[["Unnamed: 0"]]
X_test = X_test.drop("Unnamed: 0", axis=1)
print(X_test.head())
print(X_test_seg_id.head())

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(index=X_train.index, columns=X_train.columns.values.tolist(), data=X_train_scaled)
print(X_train.head())
X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(index=X_test.index, columns=X_test.columns.values.tolist(), data=X_test_scaled)
print(X_test.head())

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# create model
model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
batch_size = [64, 128, 256]
epochs = [250, 500, 1000, 1500, 2000, 2500]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['linear', 'relu']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [50, 100, 150, 200]
number_of_hd_layers = [1, 2]
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum,
                  init_mode=init_mode, activation=activation, dropout_rate=dropout_rate, dropout_rate2 = dropout_rate,
                  weight_constraint=weight_constraint, neurons=neurons, neurons2=neurons, number_of_hd_layer=number_of_hd_layers)
search = RandomizedSearchCV(estimator=model, param_distributions =param_grid, n_jobs=2, cv=5, n_iter=11,
                            scoring='neg_mean_absolute_error')
#search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3, cv=5, scoring='neg_mean_absolute_error')
search_result = search.fit(X_train, y_train)
print(search.scorer_)

print(pd.DataFrame.from_dict(search_result.cv_results_))
pd.DataFrame.from_dict(search_result.cv_results_).to_csv("./output/randomsearch.csv")
# summarize results
print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
means = search_result.cv_results_['mean_test_score']
stds = search_result.cv_results_['std_test_score']
params = search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

