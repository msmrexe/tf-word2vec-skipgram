"""
Handles downloading, preprocessing, and batching of the Text8 dataset.
"""

import gzip
import math
import random
import logging
import numpy as np
import tensorflow as tf
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

def load_data():
    """
    Downloads the Text8 dataset using gensim.
    """
    logger.info("Downloading text8 dataset...")
    try:
        text8_zip_file_path = api.load('text8', return_path=True)
        with gzip.open(text8_zip_file_path, 'rb') as file:
            file_content = file.read()
        logger.info("Dataset downloaded and unzipped successfully.")
        return file_content.decode()
    except Exception as e:
        logger.error(f"Failed to load text8 dataset: {e}")
        return None

def preprocess_text(text, threshold, min_freq):
    """
    Applies full preprocessing pipeline to the raw text:
    1. Tokenize punctuation
    2. Lowercase
    3. Remove stopwords
    4. Filter by minimum frequency
    5. Apply subsampling
    """
    if not text:
        logger.error("Received empty text for preprocessing.")
        return [], Counter()

    logger.info("Starting text preprocessing...")
    
    # 1. Replace punctuation
    text = text.replace(".", " <PERIOD> ").replace(",", " <COMMA> ")
    text = text.replace("!", " <EXCLAMATION> ").replace("?", " <QUESTION> ")
    text = text.replace(";", " <SEMICOLON> ").replace(":", " <COLON> ")
    text = text.replace("'", " <APOSTROPHE> ").replace("\"", " <QUOTE> ")

    # 2. Lowercase and split
    words = text.lower().split()
    logger.info(f"Original word count: {len(words)}")

    # 3. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    logger.info(f"Word count after stopword removal: {len(words)}")

    # 4. Remove words with frequency < min_freq
    word_counts = Counter(words)
    filtered_words = [word for word in words if word_counts[word] >= min_freq]
    logger.info(f"Word count after frequency filtering (min_freq={min_freq}): {len(filtered_words)}")

    # 5. Subsample frequent words
    total_count = len(filtered_words)
    if total_count == 0:
        logger.warning("No words left after filtering. Aborting subsampling.")
        return [], word_counts

    word_prob = {word: count / total_count for word, count in word_counts.items()}
    
    subsampled_words = [
        word for word in filtered_words
        if random.random() < (1 - math.sqrt(threshold / word_prob[word]))
    ]
    logger.info(f"Word count after subsampling (threshold={threshold}): {len(subsampled_words)}")
    
    return subsampled_words, word_counts

def create_tf_dataset(processed_words, vocab_size_limit, window_size, negative_samples, batch_size, buffer_size, train_split):
    """
    Generates skip-grams and creates tf.data.Dataset pipelines for training and testing.
    """
    logger.info("Starting dataset creation...")

    # 1. Tokenization
    tokenizer = Tokenizer(num_words=vocab_size_limit)
    tokenizer.fit_on_texts(processed_words)
    vocab_size = min(len(tokenizer.word_index) + 1, vocab_size_limit)
    logger.info(f"Vocabulary size: {vocab_size}")

    # 2. Vectorize words
    sequences = tokenizer.texts_to_sequences([processed_words])[0]
    
    # 3. Generate skip-gram pairs
    logger.info("Generating skip-gram pairs...")
    skip_gram_pairs, labels = skipgrams(
        sequences,
        vocabulary_size=vocab_size,
        window_size=window_size,
        negative_samples=negative_samples
    )
    logger.info(f"Generated {len(labels)} training pairs.")

    # 4. Extract target, context, and labels
    target_words = np.array([pair[0] for pair in skip_gram_pairs], dtype=np.int32)
    context_words = np.array([pair[1] for pair in skip_gram_pairs], dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    # 5. Split data
    sample_size = len(target_words)
    train_size = int(train_split * sample_size)
    
    logger.info(f"Splitting data: {train_size} training samples, {sample_size - train_size} validation samples.")

    train_target = target_words[:train_size]
    train_context = context_words[:train_size]
    train_labels = labels[:train_size]

    test_target = target_words[train_size:]
    test_context = context_words[train_size:]
    test_labels = labels[train_size:]

    # 6. Create TensorFlow datasets
    def create_dataset_from_slices(targets, contexts, labels):
        return tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

    train_data = create_dataset_from_slices(train_target, train_context, train_labels)
    train_data = train_data.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    test_data = create_dataset_from_slices(test_target, test_context, test_labels)
    test_data = test_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    logger.info(f"Created (train, test) batches: {len(train_data)}, {len(test_data)}")
    
    return train_data, test_data, tokenizer, vocab_size
