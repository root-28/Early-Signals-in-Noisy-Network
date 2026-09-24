import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


def build_lstm(input_shape, num_classes,
               units=64, dropout=0.2, lr=1e-3,
               feature_dim=64)
    LSTM - dropout - dense(feature_dim) - dropout - softmax.

    The dense layer named 'deep_features' is what we later reuse as the
    input to XGBoost.
    

    inp = layers.Input(shape=input_shape, name=input)

    x = layers.LSTM(units, return_sequences=False)(inp)
    x = layers.Dropout(dropout)(x)

    feat = layers.Dense(feature_dim, activation=relu,
                        name=deep_features)(x)
    x = layers.Dropout(dropout)(feat)

    out = layers.Dense(num_classes, activation=softmax, name=output)(x)

    model = Model(inp, out, name=avoa_lstm)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=sparse_categorical_crossentropy,
        metrics=[accuracy],
    )
    return model


def make_feature_extractor(lstm_model)
    Return a model that outputs the 'deep_features' activations.
    return Model(
        inputs=lstm_model.input,
        outputs=lstm_model.get_layer(deep_features).output,
        name=lstm_feature_extractor,
    )