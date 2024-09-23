import tensorflow as tf
from typing import Tuple

class SAKTModel(tf.keras.Model):
    def __init__(self, num_skills: int = 50, num_steps: int = 50, hidden_units: int = 200, dropout_rate: float = 0.2, num_heads: int = 2):
        """
        Custom neural network model using multi-head attention for sequence prediction tasks.

        Args:
            num_skills (int): The number of unique skills (or problems) in the dataset.
            num_steps (int): The number of steps (sequence length) for the model.
            hidden_units (int): The number of hidden units in the attention and feed-forward layers.
            dropout_rate (float): The dropout rate applied to prevent overfitting.
            num_heads (int): The number of attention heads used in the MultiHeadAttention layer.
        """
        super(SAKTModel, self).__init__()
        self.batch_size = 128
        self.num_skills = num_skills
        self.num_steps = num_steps
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.enc_embedding = tf.keras.layers.Embedding(input_dim=num_skills * 2, output_dim=hidden_units)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=num_steps, output_dim=hidden_units)
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(self.hidden_units)
        ])
        self.sigmoid_w = tf.keras.layers.Dense(num_skills, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_2=0.98)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]): 
                A tuple containing input tensors for problem IDs, skill IDs, target IDs, and target correctness.
            training (bool): A boolean flag indicating if the model is in training mode.
        
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: 
                - Predictions (logits) for each step in the sequence.
                - Loss value for the current batch.
                - Attention weights from the MultiHeadAttention layer.
        """
        x, problems, target_id, target_correctness = inputs
        x = tf.cast(x, tf.int32)
        key_masks = tf.cast(tf.not_equal(x, 0), tf.float32)
        enc = self.enc_embedding(x) + self.pos_embedding(tf.range(self.num_steps - 1))
        enc = enc * tf.expand_dims(key_masks, axis=-1)
        enc = self.dropout(enc, training=training)
        attn_output, attention_weights = self.multihead_attention(enc, enc, training=training, return_attention_scores=True)
        attn_output += enc
        attn_output = tf.keras.layers.LayerNormalization()(attn_output)
        output = self.feed_forward(attn_output)
        output += attn_output
        output = tf.keras.layers.LayerNormalization()(output)
        output = tf.reshape(output, [-1, self.hidden_units])
        logits = self.sigmoid_w(output)
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, target_id)
        pred = tf.sigmoid(selected_logits)
        target_correctness = tf.cast(target_correctness, tf.float32)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))
        return pred, loss, attention_weights
