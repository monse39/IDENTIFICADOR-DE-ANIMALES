import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# 1. Configuracion del programa
ruta = "banco_imagenes"
tamano = (224, 224)
lote = 16

print("IDENTIFICADOR DE ANIMALES Y OBJETOS")
print("Preparando banco de imagenes...")

# 2. Cargar el banco de imagenes
entrenamiento = tf.keras.utils.image_dataset_from_directory(
    ruta,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=tamano,
    batch_size=lote
)

validacion = tf.keras.utils.image_dataset_from_directory(
    ruta,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=tamano,
    batch_size=lote
)

clases = entrenamiento.class_names

print("Categorias encontradas:", clases)

# 3. Preparar las imagenes
entrenamiento = entrenamiento.map(
    lambda x, y: (preprocess_input(x), y)
)

validacion = validacion.map(
    lambda x, y: (preprocess_input(x), y)
)

# 4. Crear el modelo de inteligencia artificial
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base.trainable = False

modelo = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(clases), activation="softmax")
])

# 5. Configurar el entrenamiento
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Entrenar el modelo
print("\nEntrenando inteligencia artificial...")

modelo.fit(
    entrenamiento,
    validation_data=validacion,
    epochs=8
)

# 7. Guardar el modelo
modelo.save("modelo_animales.keras")

print("\nEntrenamiento terminado.")

# 8. Identificar una nueva imagen
while True:

    ruta_imagen = input(
        "\nEscribe la ruta de una imagen o SALIR: "
    )

    if ruta_imagen.lower() == "salir":
        print("Programa finalizado.")
        break

    try:
        img = image.load_img(
            ruta_imagen,
            target_size=tamano
        )

        datos = image.img_to_array(img)
        datos = np.expand_dims(datos, axis=0)
        datos = preprocess_input(datos)

        # Realizar prediccion
        resultado = modelo.predict(datos, verbose=0)

        posicion = np.argmax(resultado[0])
        confianza = resultado[0][posicion]

        print("\nRESULTADO DEL ANALISIS")

        if confianza < 0.65:
            print("No se pudo identificar con suficiente confianza.")
        else:
            print("La imagen corresponde a:", clases[posicion])
            print("Precision estimada:",
                  round(float(confianza) * 100, 2), "%")

    except Exception as error:
        print("Error al procesar la imagen:", error)