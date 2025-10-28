"""
Utility functions for logging and exporting embeddings.
"""

import os
import sys
import logging
import tensorflow as tf

def setup_logging(log_file):
    """
    Configures the root logger to output to both console and a log file.
    """
    try:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Set up formatting
        log_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

        logging.info("Logging configured successfully.")
        return root_logger

    except Exception as e:
        print(f"Error setting up logger: {e}")
        sys.exit(1)

def save_embeddings_for_projector(embeddings, tokenizer, output_dir):
    """
    Saves the learned embeddings and metadata in TSV format
    for the TensorFlow Embedding Projector.

    Args:
        embeddings (np.ndarray): The embedding matrix.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The fitted tokenizer.
        output_dir (str): Directory to save the .tsv files.
    """
    vecs_path = os.path.join(output_dir, "vecs.tsv")
    meta_path = os.path.join(output_dir, "meta.tsv")

    try:
        with open(vecs_path, "w", encoding="utf-8") as vecs_file, \
             open(meta_path, "w", encoding="utf-8") as meta_file:
            
            for word, idx in tokenizer.word_index.items():
                if idx >= embeddings.shape[0]:
                    logging.warning(f"Word '{word}' index {idx} out of bounds for embeddings shape {embeddings.shape}")
                    continue
                
                vec = embeddings[idx]
                
                # Write metadata (word)
                meta_file.write(f"{word}\n")
                
                # Write vector
                vecs_file.write("\t".join([str(x) for x in vec]) + "\n")
        
        logging.info(f"Embeddings saved to {vecs_path}")
        logging.info(f"Metadata saved to {meta_path}")

    except IOError as e:
        logging.error(f"Error writing embedding files: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during embedding export: {e}")
