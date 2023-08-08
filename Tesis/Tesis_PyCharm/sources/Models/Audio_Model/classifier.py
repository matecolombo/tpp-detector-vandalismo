import tensorflow as tf

def parse_sequence_example(example_proto):
    # Define the feature description for parsing
    context_feature_description = {
        'video_id': tf.io.FixedLenFeature([], tf.string),
        'start_time_seconds': tf.io.FixedLenFeature([], tf.float32),
        'end_time_seconds': tf.io.FixedLenFeature([], tf.float32),
        'labels': tf.io.VarLenFeature(tf.int64),
    }
    sequence_feature_description = {
        'audio_embedding': tf.io.VarLenFeature(tf.string),
    }

    # Parse the sequence example
    context, sequence = tf.io.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_feature_description,
        sequence_features=sequence_feature_description
    )

    # Get the 'audio_embedding' feature and convert it to a dense tensor
    audio_embedding = tf.sparse.to_dense(sequence['audio_embedding'], default_value="")
    
    # Get the 'labels' and determine if it's a "grito" or "no grito" based on the label values
    labels = tf.sparse.to_dense(context['labels']).numpy()
    is_grito = any(label in labels for label in [1, 522])  # Assuming 1 and 522 correspond to "grito"

    return audio_embedding, is_grito

# Load the .tfrecord file
file_path = "path_to_your_file.tfrecord"
dataset = tf.data.TFRecordDataset(file_path)

# Map the parsing function to the dataset
dataset = dataset.map(parse_sequence_example)

# Now you can iterate through the dataset to get the audio embeddings and labels
for audio_embedding, is_grito in dataset:
    # Here, 'audio_embedding' is the audio feature, and 'is_grito' is a boolean indicating whether it's a "grito" or not.
    if is_grito:
        print("It's a grito!")
    else:
        print("It's not a grito.")


