import tensorflow as tf
from typing import List, Tuple

class SAKTModel(tf.keras.Model):
    def __init__(self, num_skills: int, num_problems: int, num_steps: int, d_model: int = 128, num_heads: int = 4, dropout_rate: float = 0.1):
        super(SAKTModel, self).__init__()
        self.num_skills = num_skills
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.d_model = d_model

        self.skill_embedding = tf.keras.layers.Embedding(input_dim=num_skills+1, output_dim=d_model)
        self.problem_embedding = tf.keras.layers.Embedding(input_dim=num_problems+1, output_dim=d_model)
        self.correctness_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=d_model)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=num_steps, output_dim=d_model)
        
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        skills, problems, prev_responses = inputs
        
        batch_size = tf.shape(skills)[0]
        positions = tf.range(start=0, limit=self.num_steps-1, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [batch_size, 1])
        
        skill_embed = self.skill_embedding(skills[:, 1:])  # Shift by 1 to align with responses
        problem_embed = self.problem_embedding(problems[:, 1:])
        correctness_embed = self.correctness_embedding(prev_responses)
        pos_embed = self.pos_embedding(positions)
        
        x = skill_embed + problem_embed + correctness_embed + pos_embed
        
        attention_mask = tf.linalg.band_part(tf.ones((self.num_steps-1, self.num_steps-1)), -1, 0)
        attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 0), 0)
        
        attn_output = self.multi_head_attention(query=x, key=x, value=x, attention_mask=attention_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        
        output = self.output_layer(out2)
        return output[:, :, 0]

@tf.function
def train_step(model: SAKTModel, x_skills: tf.Tensor, x_problems: tf.Tensor, x_responses: tf.Tensor, y: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer, loss_fn: tf.keras.losses.Loss) -> Tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        predictions = model([x_skills, x_problems, x_responses], training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

@tf.function
def val_step(model: SAKTModel, x_skills: tf.Tensor, x_problems: tf.Tensor, x_responses: tf.Tensor, y: tf.Tensor, loss_fn: tf.keras.losses.Loss) -> Tuple[tf.Tensor, tf.Tensor]:
    predictions = model([x_skills, x_problems, x_responses], training=False)
    loss = loss_fn(y, predictions)
    return loss, predictions