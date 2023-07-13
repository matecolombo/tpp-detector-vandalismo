
import os
import numpy as np
import matplotlib.pyplot as plt

from Preprocess_webcam import filename_date


def alerta_violencia():
    print("Alerta: violencia en el pabellón.")


def print_prediction(prediction):
    # Crear la figura y los ejes
    fig, ax = plt.subplots()
    ax.get_xaxis().set_visible(False)

    prediction_dir = '../Preprocess/Video_Webcam/Predictions'
    # Generar los índices para las barras
    indices = np.arange(1)

    # Ancho de las barras
    ancho_barras = 0.20
    umbral_violencia = 0.8

    # Crear las barras iniciales (vacías)
    barras1 = ax.bar(indices, prediction.shape[0], ancho_barras, label='Pelea', color='tab:red')
    barras2 = ax.bar(indices + ancho_barras * 4 / 3, prediction.shape[1], ancho_barras, label='No pelea',
                     color='tab:blue')
    ax.axhline(umbral_violencia, color='black', linestyle='--')
    # Agregar etiquetas a los ejes

    ax.set_ylabel('Probabilidad')

    # Agregar título al gráfico
    ax.set_title('Evolución de la probabilidad de violencia')

    # Agregar leyenda
    ax.legend()

    plt.ylim([0, 1])

    # Actualizar las barras con cada actualización
    for i in range(prediction.shape[0]):
        barras1[0].set_height(prediction[i, 0])
        barras2[0].set_height(prediction[i, 1])
        file_name = filename_date()
        prediction_path = os.path.join(prediction_dir, file_name)
        plt.savefig(prediction_path)
        plt.pause(1)  # Pausa de 0.5 segundos entre cada actualización
        plt.draw()

        if prediction[i, 0] >= umbral_violencia:
            alerta_violencia()

    # Mostrar el gráfico final
    #plt.show()
    plt.close()


'''
# Definir la matriz "prediction" con las actualizaciones (ejemplo con 10 actualizaciones)
prediction = np.array([[0.2971856, 0.7028144],
                       [0.12679586, 0.8732041],
                       [0.10911757, 0.89088243],
                       [0.61735284, 0.38264713],
                       [0.66615975, 0.33384025],
                       [0.15630996, 0.8436901]])

print_prediction(prediction)
'''
