"""
Configuration file for model hyperparameters and pipeline settings.
"""

# Data settings
VOCAB_SIZE_LIMIT = 30000
MIN_WORD_FREQUENCY = 5
SUBSAMPLING_THRESHOLD = 1e-5

# Model settings
EMBEDDING_DIM = 128

# Training settings
EPOCHS = 5
BATCH_SIZE = 2048
BUFFER_SIZE = 10000
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.9

# Skip-gram generation settings
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 1.0

# Paths
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
LOG_FILE = f"{LOG_DIR}/training.log"
OUTPUT_DIR = "./data"
