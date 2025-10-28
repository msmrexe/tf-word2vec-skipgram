"""
Contains the custom training and evaluation loops for the SkipGram model.
"""

import time
import logging
import tensorflow as tf
import os

logger = logging.getLogger(__name__)

def train_model(model, train_data, test_data, epochs, learning_rate, checkpoint_dir):
    """
    Runs the custom training loop for the Skip-Gram model.

    Args:
        model (tf.keras.Model): The SkipGramModel instance.
        train_data (tf.data.Dataset): Training dataset.
        test_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.
        learning_rate (float): The optimizer learning rate.
        checkpoint_dir (str): Directory to save model checkpoints.

    Returns:
        tuple: (trained_model, history_dict)
    """

    # 1. Initialize optimizer, loss, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    # 2. Set up checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5
    )
    
    history = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": []
    }

    # 3. Define train and test steps
    @tf.function
    def train_step(target, context, labels):
        with tf.GradientTape() as tape:
            predictions = model((target, context), training=True)
            loss = loss_fn(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        train_acc_metric.update_state(labels, predictions)
        return loss

    @tf.function
    def test_step(target, context, labels):
        predictions = model((target, context), training=False)
        loss = loss_fn(labels, predictions)
        val_acc_metric.update_state(labels, predictions)
        return loss

    # 4. The training loop
    logger.info("Starting model training...")
    for epoch in range(epochs):
        logger.info(f"--- Starting epoch: {epoch+1}/{epochs} ---")
        start_time = time.time()

        # Training
        train_loss = 0.0
        for batch_index, ((target_batch, context_batch), label_batch) in enumerate(train_data):
            loss_val = train_step(target_batch, context_batch, label_batch)
            train_loss += loss_val

            if batch_index % 5000 == 0 and batch_index > 0:
                logger.info(f"  Epoch {epoch+1}, Batch {batch_index}: Loss={loss_val:.4f}")

        train_acc = train_acc_metric.result()
        avg_train_loss = train_loss / len(train_data)
        logger.info(f"Training acc over epoch: {train_acc:.4f}")
        logger.info(f"Cumulative train loss: {train_loss.numpy():.4f}")

        # Validation
        test_loss = 0.0
        for (target_batch, context_batch), label_batch in test_data:
            test_loss += test_step(target_batch, context_batch, label_batch)
        
        val_acc = val_acc_metric.result()
        avg_test_loss = test_loss / len(test_data)
        logger.info(f"Validation acc over epoch: {val_acc:.4f}")
        logger.info(f"Cumulative test loss: {test_loss.numpy():.4f}")

        end_time = time.time()
        logger.info(f"Time taken for epoch: {end_time - start_time:.2f}s")

        # Save metrics and reset states
        history["train_loss"].append(avg_train_loss.numpy())
        history["test_loss"].append(avg_test_loss.numpy())
        history["train_acc"].append(train_acc.numpy())
        history["val_acc"].append(val_acc.numpy())

        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        # Save checkpoint
        save_path = manager.save()
        logger.info(f"Saved checkpoint for epoch {epoch+1} at {save_path}")

    logger.info("Training finished.")
    return model, history
