from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model(optimizer='adagrad',
                 kernel_initializer='glorot_uniform', 
                 dropout=0.2):
    # model = Sequential()
    # model.add(Dense(100,activation='relu',kernel_initializer=kernel_initializer, kernel_regularizer = tf.keras.regularizers.l1()))
    # model.add(Dropout(dropout))
    # model.add(Dense(100,activation='relu',kernel_initializer=kernel_initializer, kernel_regularizer = tf.keras.regularizers.l1()))
    # model.add(Dense(1,activation='sigmoid',kernel_initializer=kernel_initializer))

    # model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    ############################################
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation = 'relu',
                                kernel_initializer='random_normal',
                                bias_initializer='random_normal',
                                kernel_regularizer = tf.keras.regularizers.l1_l2()))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr = 0.001)

    model.compile(optimizer = optimizer,
                loss = loss_fn,
                metrics = ["accuracy"])
    ############################################
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(2, activation = 'relu',
    #                             kernel_initializer='random_normal',
    #                             bias_initializer='random_normal'))

    # model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    # loss_fn = tf.keras.losses.BinaryCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(lr = 0.001)

    # model.compile(optimizer = optimizer,
    #             loss = loss_fn,
    #             metrics = ["accuracy"])
    return model


def sklearn_pipeline():
    lrreduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.7,
        patience=30,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=400,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # wrap the model using the function you created
    clf = KerasClassifier(build_fn=create_model,
                            verbose=0,
                            epochs = 400,
                            validation_split = 0.2)

    # just create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',clf)
    ])

    return pipeline
    # pipeline.fit(X_train, y_train)

