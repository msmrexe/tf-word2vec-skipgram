"""
Defines the Skip-Gram model architecture using TensorFlow/Keras.
"""

import tensorflow as tf

class SkipGramModel(tf.keras.Model):
    """
    Implementation of the Skip-Gram model.
    
    This model uses two separate embedding layers:
    1. Target Embedding: For the center word.
    2. Context Embedding: For the context words (positive and negative).
    
    The forward pass computes the dot product between the target and context
    embeddings, which is then passed to a sigmoid cross-entropy loss.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the model layers.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        super(SkipGramModel, self).__init__()
        
        # Embedding layer for the target (center) word
        self.target_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim,
            name="target_embedding"
        )
        
        # Embedding layer for the context (surrounding/negative) word
        self.context_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim,
            name="context_embedding"
        )
        
        # Dot product layer to compute similarity
        self.dot_product = tf.keras.layers.Dot(axes=-1)
        
        # Flatten the output for the loss function
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        """
        Defines the forward pass of the model.

        Args:
            inputs (tuple): A tuple containing (target_word, context_word)
                            tensors, each of shape (batch_size,).

        Returns:
            tf.Tensor: The logits (dot product) of shape (batch_size, 1).
        """
        target, context = inputs
        
        # Look up embeddings
        target_embed = self.target_embedding(target)  # (batch_size, embedding_dim)
        context_embed = self.context_embedding(context) # (batch_size, embedding_dim)
        
        # Compute dot product
        dot_product = self.dot_product([target_embed, context_embed]) # (batch_size, 1)
        
        # Flatten for loss calculation
        output = self.flatten(dot_product) # (batch_size,)
        return output
