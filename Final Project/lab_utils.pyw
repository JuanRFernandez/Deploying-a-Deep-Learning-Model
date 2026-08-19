import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


def df_to_tfdata(df, topic_lookup, title_tokenizer, batch_size=32, buffer_size=1000, shuffle=False):
    '''Converts a pandas dataframe to a tf.data.Dataset'''

    # Extract the news titles and topics
    inputs = df['title']    
    labels = df['topic']

    # Convert the titles and topics to integers
    sequences = title_tokenizer(inputs)
    labels = topic_lookup(labels)

    # Combine the numeric representations to a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((sequences,labels))

    # Shuffle and create batches
    if shuffle:
        tf_dataset = dataset.shuffle(buffer_size).batch(batch_size)
    else:
        tf_dataset = dataset.batch(batch_size)

    return tf_dataset


def model_reset_weights(model):
    '''Resets the model with random weights'''

    # Loop through the layers of the model
    for ix, layer in enumerate(model.layers):

        # Reset layers with kernel and bias initializers
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer
    
            old_weights, old_biases = model.layers[ix].get_weights()
    
            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=old_biases.shape)])

        # Reset layers with an embedding initializer
        if hasattr(model.layers[ix], 'embeddings_initializer'):
            embeddings_initializer = model.layers[ix].embeddings_initializer
    
            embedding_weights = model.layers[ix].get_weights()[0]
    
            model.layers[ix].set_weights([
                embeddings_initializer(shape=embedding_weights.shape)])
    
    return model

def get_errors(model, df, tokenizer, topic_lookup, topic, num_items=20):
    '''Prints erroneous predictions for a given a news topic'''

    # Convert news titles to integer sequences
    df_title_np = df['title'].to_numpy()
    df_title_np_tokenized = tokenizer(df_title_np)

    # Convert news topics to integers
    df_labels_np = df['topic'].to_numpy()
    
    # Get the list of topics
    topics = topic_lookup.get_vocabulary()

    # Pass the news titles to the model and get predictions
    predictions = model.predict([df_title_np_tokenized], verbose=0)

    # Get the top predictions
    top_predictions = map(np.argmax, predictions)

    # Get the ground truth
    ground_truth = topic_lookup(df_labels_np)

    # Initialize counter
    count = 0

    # Loop through each pair of prediction and ground truth
    for index, (result, gt) in enumerate(zip(top_predictions, ground_truth)):
        
        # For a given class, print if the prediction does not match the ground truth
        if (topics[result]==topic) and (result != gt):
            print(
                f'label: {topics[gt]}\n'
                f'prediction: {topics[result]}\n'
                f'title: {df_title_np[index]}\n'
            )
            
            count+=1

        # Stop if we get a specified number of items
        if count == num_items:
            break



def save_data(df, data_dir, filename):
    '''Saves a dataframe to a given directory as a CSV file'''
    
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(f'{data_dir}/{filename}', index=False)

def save_vocab(tokenizer, vocab_dir):
    '''Saves a vocabulary to a given directory as a text file''' 
    
    os.makedirs(vocab_dir, exist_ok=True)
    tokenizer.save_assets(vocab_dir)

def save_labels(topic_lookup, vocab_dir):
    '''Saves a labels list to a given directory as a text file''' 
    
    os.makedirs(vocab_dir, exist_ok=True)
    
    labels = topic_lookup.get_vocabulary()
    labels_filepath = f'{vocab_dir}/labels.txt'
    
    with open(labels_filepath, "w") as f:
        f.write("\n".join([str(w) for w in labels]))

    
def set_experiment_dirs(base_dir):
    '''Sets the data, model, and vocab directories for an experiment''' 

    data_dir = f'{base_dir}/data'
    model_dir = f'{base_dir}/model'
    vocab_dir = f'{base_dir}/vocab'
    
    return data_dir, model_dir, vocab_dir

def print_metric_per_topic(df, topics, topic_lookup, title_preprocessor, model):
    '''Prints the accuracy per class of news topics''' 

    print(f'ACCURACY PER TOPIC:\n')
    
    for topic in topics:
        topic_df = df[df.topic == topic]
        topic_ds = df_to_tfdata(topic_df, topic_lookup, title_preprocessor)
        result = model.evaluate(topic_ds, verbose=0)
        accuracy = result[1] * 100
        accuracy = "%.2f " % accuracy
        print(f'{topic}: {accuracy}')