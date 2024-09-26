import time
from typing import Tuple, Generator, List
import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


class SAKTModel(tf.keras.Model):
    def __init__(self, file_path: str, num_steps: int = 100, batch_size: int = 32, d_model: int = 256,
                 num_heads: int = 8, dropout_rate: float = 0.2):
        super(SAKTModel, self).__init__()
        self.file_path = file_path
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.exercise_map = {}
        self.num_exercises = 0
        self.exercise_embedding = None
        self.positional_embedding = None
        self.multi_head_attention = None
        self.dropout1 = None
        self.layer_norm1 = None
        self.ffn = None
        self.dropout2 = None
        self.layer_norm2 = None
        self.output_layer = None

    def preprocess(self) -> None:
        df = pd.read_csv(self.file_path)
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df[['user_xid', 'old_problem_id', 'discrete_score', 'start_time']]
        df = df.sort_values(by=['user_xid', 'start_time'])
        self.exercise_map = {problem: idx for idx, problem in enumerate(df['old_problem_id'].unique(), start=1)}
        self.num_exercises = len(self.exercise_map)

        self.exercise_embedding = tf.keras.layers.Embedding(input_dim=self.num_exercises + 1, output_dim=self.d_model)
        self.positional_embedding = tf.keras.layers.Embedding(input_dim=self.num_steps, output_dim=self.d_model)

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model,
                                                                       dropout=self.dropout_rate)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_model * 4, activation='relu'),
            tf.keras.layers.Dense(self.d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        exercises, responses = inputs

        # Create position IDs
        positions = tf.range(start=0, limit=tf.shape(exercises)[1], delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [tf.shape(exercises)[0], 1])

        # Get embeddings
        exercise_embeddings = self.exercise_embedding(exercises)
        positional_embeddings = self.positional_embedding(positions)

        # Combine exercise and response embeddings
        interaction_embeddings = tf.concat([exercise_embeddings, tf.expand_dims(tf.cast(responses, tf.float32), -1)],
                                           axis=-1)

        # Add positional encodings
        interaction_embeddings += positional_embeddings

        # Self-attention
        attention_output = self.multi_head_attention(
            query=exercise_embeddings,
            key=interaction_embeddings,
            value=interaction_embeddings,
            attention_mask=self.create_look_ahead_mask(tf.shape(exercises)[1]))

        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layer_norm1(attention_output + exercise_embeddings)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(ffn_output + out1)

        # Output layer
        output = self.output_layer(out2)
        return output

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def fit(self, num_epochs: int = 10):
        # Create data generator
        train_data = self._data_generator()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_auc = tf.keras.metrics.AUC(name='train_auc')
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        for epoch in range(num_epochs):
            train_loss.reset_state()
            train_auc.reset_state()
            train_accuracy.reset_state()

            train_pbar = tqdm.tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=100)
            for batch in train_pbar:
                x_exercises_batch, x_responses_batch, y_batch = batch
                loss, predictions = self._train_step(self, x_exercises_batch, x_responses_batch, y_batch, optimizer,
                                                     loss_fn)

                train_loss.update_state(loss)
                train_auc.update_state(y_batch, predictions)
                train_accuracy.update_state(y_batch, predictions)

                train_pbar.set_postfix({
                    'loss': f'{train_loss.result():.4f}',
                    'auc': f'{train_auc.result():.4f}',
                    'acc': f'{train_accuracy.result():.4f}'
                })

    def evaluate(self) -> None:
        val_data = self._data_generator()
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_auc = tf.keras.metrics.AUC(name='val_auc')
        val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

        val_pbar = tqdm.tqdm(val_data, desc=f"Evaluation [Val]", ncols=100)
        for batch in val_pbar:
            x_exercises_batch, x_responses_batch, y_batch = batch
            loss, predictions = self._val_step(self, x_exercises_batch, x_responses_batch, y_batch,
                                               tf.keras.losses.BinaryCrossentropy())

            val_loss.update_state(loss)
            val_auc.update_state(y_batch, predictions)
            val_accuracy.update_state(y_batch, predictions)

            val_pbar.set_postfix({
                'loss': f'{val_loss.result():.4f}',
                'auc': f'{val_auc.result():.4f}',
                'acc': f'{val_accuracy.result():.4f}'
            })

    def _data_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        df = pd.read_csv(self.file_path)
        user_ids = df['user_xid'].unique()
        num_users = len(user_ids)

        while True:
            np.random.shuffle(user_ids)
            for start_idx in range(0, num_users, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_users)
                batch_user_ids = user_ids[start_idx:end_idx]

                batch_exercises, batch_responses, batch_targets = [], [], []

                for user_id in batch_user_ids:
                    user_data = df[df['user_xid'] == user_id].sort_values('start_time')
                    exercise_seq = [self.exercise_map[problem] for problem in user_data['old_problem_id']]
                    response_seq = user_data['discrete_score'].tolist()

                    if len(exercise_seq) > self.num_steps:
                        start_index = np.random.randint(0, len(exercise_seq) - self.num_steps)
                        exercise_seq = exercise_seq[start_index:start_index + self.num_steps]
                        response_seq = response_seq[start_index:start_index + self.num_steps]
                    else:
                        padding = self.num_steps - len(exercise_seq)
                        exercise_seq = [0] * padding + exercise_seq
                        response_seq = [0] * padding + response_seq

                    batch_exercises.append(exercise_seq)
                    batch_responses.append(response_seq[:-1])  # All but last
                    batch_targets.append(response_seq[1:])  # All but first

                yield (np.array(batch_exercises, dtype=np.int32),
                       np.array(batch_responses, dtype=np.int32),
                       np.array(batch_targets, dtype=np.float32))

    @staticmethod
    @tf.function
    def _train_step(model, x_exercises, x_responses, y, optimizer, loss_fn) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            predictions = model([x_exercises, x_responses], training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, predictions

    @staticmethod
    @tf.function
    def _val_step(model, x_exercises, x_responses, y, loss_fn) -> Tuple[tf.Tensor, tf.Tensor]:
        predictions = model([x_exercises, x_responses], training=False)
        loss = loss_fn(y, predictions)
        return loss, predictions