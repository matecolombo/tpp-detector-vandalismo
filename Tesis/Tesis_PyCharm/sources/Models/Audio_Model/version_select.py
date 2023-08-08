## tf 3.7
model_file = '../../Models/keras_model.h5' 
modelo_padre = load_model(model_file, compile=False)
modelo_padre.save('./modelo_padre', save_format='tf')

## tf 3.10
modelo_padre = load_model('./modelo_padre')




## tf 2.11
