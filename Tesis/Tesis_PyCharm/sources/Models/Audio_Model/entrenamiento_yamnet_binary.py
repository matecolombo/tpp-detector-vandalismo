import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from yamnet import pad_waveform, waveform_to_log_mel_spectrogram_patches


def yamnet_binary(features, params):
    """Define a binary version of YAMNet for scream vs non-scream classification."""
    net = layers.Reshape(
        (params.patch_frames, params.patch_bands, 1),
        input_shape=(params.patch_frames, params.patch_bands))(features)
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
    embeddings = layers.GlobalAveragePooling2D()(net)
    logits = layers.Dense(units=1, activation='sigmoid', use_bias=True)(embeddings)  # Binary classification
    return logits


def yamnet_scream_detection_model(params):
    """Defines the YAMNet-based scream detection model.

    Args:
        params: An instance of Params containing hyperparameters.

    Returns:
        A binary classification model accepting (num_samples,) waveform input and emitting:
        - logits: (1,) sigmoid output indicating the probability of a scream
        - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
        - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
    """
    waveform = layers.Input(batch_shape=(None,), dtype=tf.float32)
    waveform_padded = pad_waveform(waveform, params)
    log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(
        waveform_padded, params)
    logits = yamnet_binary(features, params)
    detection_model = tf.keras.Model(name='yamnet_scream_detection', inputs=waveform,
        outputs=[logits, features, log_mel_spectrogram])
    return detection_model



# ------------- Define los hiperparámetros para el modelo y el entrenamiento -------------
class Params:
    patch_frames = 96
    patch_bands = 64
    num_classes = 1  # Binary classification
    classifier_activation = 'sigmoid'
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

params = Params()

# ------------- Normaliza los datos de entrada para que estén en el rango [0, 1] -------------
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# ------------- Convierte las etiquetas a un formato adecuado para la clasificación binaria (0 para "no gritos" y 1 para "gritos") -------------
y_train_binary = (y_train == "gritos").astype(int)
y_test_binary = (y_test == "gritos").astype(int)


# -------------------------- Construye el modelo ---------------------------------------
model = yamnet_scream_detection_model(params)

# Compila el modelo con la función de pérdida y métricas apropiadas para la clasificación binaria
model.compile(optimizer=Adam(learning_rate=params.learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ------------- Entrenamiento el modelo en los datos de entrenamiento -------------
model.fit(X_train, y_train_binary, batch_size=params.batch_size, epochs=params.num_epochs, validation_split=0.1)


# ------------- Evalúa el modelo en los datos de prueba -------------
loss, accuracy = model.evaluate(X_test, y_test_binary)
print("Test accuracy:", accuracy)


# ------------- Realiza predicciones en nuevos datos -------------
new_data = ...  # Carga o graba un nuevo audio para probar el modelo
normalized_data = new_data / np.max(new_data)
prediction = model.predict(np.expand_dims(normalized_data, axis=0))
if prediction[0, 0] >= 0.5:
    print("Es un grito.")
else:
    print("No es un grito.")
