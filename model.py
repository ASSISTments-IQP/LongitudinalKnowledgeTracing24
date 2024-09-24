import tensorflow as tf
import numpy as np

class SAKTModel(tf.keras.Model):
    def __init__(self, num_skills: int, num_steps: int, d_model: int = 200, num_heads: int = 8, dropout_rate: float = 0.2):
        """
        SAKTModel based on the self-attentive architecture for knowledge tracing.
        """
        super(SAKTModel, self).__init__()
        self.num_skills = num_skills
        self.num_steps = num_steps
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Embedding layers for the interaction and positional encoding
        self.interaction_embedding = tf.keras.layers.Embedding(input_dim=2 * num_skills, output_dim=d_model)
        self.exercise_embedding = tf.keras.layers.Embedding(input_dim=num_skills, output_dim=d_model)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=num_steps, output_dim=d_model)
        
        # Multi-head attention mechanism
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        
        # Feed-forward network
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Dropout and layer normalization
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Output prediction layer (binary classification for correctness)
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x, problems, target_ids, target_correctness = inputs
        
        # Interaction embedding and positional encoding
        interaction_emb = self.interaction_embedding(x)
        pos_emb = self.pos_embedding(tf.range(self.num_steps - 1))
        interaction_emb += pos_emb
        
        # Multi-head attention: queries (problems) and keys/values (interaction embedding)
        attention_output, attention_weights = self.attention(
            query=self.exercise_embedding(problems),
            key=interaction_emb,
            value=interaction_emb,
            return_attention_scores=True
        )
        
        # Residual connection and layer normalization after attention
        attention_output = self.layer_norm1(attention_output + interaction_emb)
        
        # Feed-forward layer with residual connection and layer normalization
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.layer_norm2(feed_forward_output + attention_output)
        
        # Reshape the feed-forward output for the final prediction
        logits = self.final_layer(tf.reshape(feed_forward_output, [-1, self.d_model]))
        logits = tf.reshape(logits, [-1])
        
        # Gather predictions corresponding to the target problem IDs
        selected_logits = tf.gather(logits, target_ids)
        pred = tf.sigmoid(selected_logits)
        
        # Compute loss for the given target correctness
        target_correctness = tf.cast(target_correctness, tf.float32)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))
        
        return pred, loss, attention_weights
