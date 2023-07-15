from tflite_support.task import audio

classifier = audio.AudioClassifier.create_from_file('lite-model_yamnet_tflite_1.tflite')

tensor = audio.TensorAudio.create_from_file('audio.wav', classifier.required_input_buffr_size)

result = classifier.classify(tensor)
