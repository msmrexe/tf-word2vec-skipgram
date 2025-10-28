"""
Main entry point for running the Word2Vec Skip-Gram pipeline.
"""

import os
import argparse
import logging
import tensorflow as tf
from src import utils, config, data_loader, model, train

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def main(args):
    """
    Runs the full data processing and model training pipeline.
    """
    # 1. Setup
    utils.setup_logging(config.LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info("Starting Word2Vec Skip-Gram pipeline...")
    
    # Create necessary directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Log arguments
    logger.info("Running with the following arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # 2. Load and Preprocess Data
    raw_text = data_loader.load_data()
    if not raw_text:
        logger.error("Failed to load data. Exiting.")
        return

    processed_words, _ = data_loader.preprocess_text(
        raw_text,
        args.subsampling_threshold,
        args.min_word_frequency
    )
    if not processed_words:
        logger.error("No words remaining after preprocessing. Exiting.")
        return

    # 3. Create TF Datasets
    train_ds, test_ds, tokenizer, vocab_size = data_loader.create_tf_dataset(
        processed_words,
        args.vocab_size_limit,
        args.window_size,
        args.negative_samples,
        args.batch_size,
        config.BUFFER_SIZE,
        config.TRAIN_SPLIT
    )

    # 4. Initialize Model
    logger.info(f"Initializing SkipGramModel with vocab_size={vocab_size} and embedding_dim={args.embedding_dim}")
    skipgram_model = model.SkipGramModel(vocab_size, args.embedding_dim)

    # 5. Train Model
    trained_model, history = train.train_model(
        skipgram_model,
        train_ds,
        test_ds,
        args.epochs,
        args.learning_rate,
        args.checkpoint_dir
    )

    logger.info("Training complete.")
    logger.info(f"Final training history: {history}")

    # 6. Save Embeddings
    logger.info("Extracting and saving final embeddings for projector...")
    embeddings = trained_model.target_embedding.get_weights()[0]
    
    utils.save_embeddings_for_projector(
        embeddings,
        tokenizer,
        args.output_dir
    )

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Word2Vec Skip-Gram model.")
    
    # Model Hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=config.EMBEDDING_DIM,
                        help="Dimensionality of the word embeddings.")
    parser.add_argument("--vocab_size_limit", type=int, default=config.VOCAB_SIZE_LIMIT,
                        help="Maximum number of words in the vocabulary.")
    parser.add_argument("--window_size", type=int, default=config.WINDOW_SIZE,
                        help="Context window size for skip-grams.")
    parser.add_argument("--negative_samples", type=float, default=config.NEGATIVE_SAMPLES,
                        help="Number of negative samples to use.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE,
                        help="Optimizer learning rate.")

    # Preprocessing
    parser.add_argument("--min_word_frequency", type=int, default=config.MIN_WORD_FREQUENCY,
                        help="Minimum word frequency to be included in vocabulary.")
    parser.add_argument("--subsampling_threshold", type=float, default=config.SUBSAMPLING_THRESHOLD,
                        help="Threshold for frequent word subsampling.")

    # Paths
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Directory to save embedding projector files.")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory to save model checkpoints.")

    args = parser.parse_args()
    main(args)
