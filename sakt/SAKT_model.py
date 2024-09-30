import tensorflow as tf
from typing import Tuple, Generator, Dict
import pandas as pd
import numpy as np
import tqdm


class SAKTModel(tf.keras.Model):
    def __init__(self, num_steps: int = 50, batch_size: int = 32, d_model: int = 128,
                 num_heads: int = 8, dropout_rate: float = 0.2):
        super(SAKTModel, self).__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.exercise_map = {}
        self.num_exercises = 0
        self.exercise_embedding = None
        self.response_embedding = None
        self.position_embedding = None
        self.multi_head_attention = None
        self.dropout1 = None
        self.layer_norm1 = None
        self.ffn = None
        self.dropout2 = None
        self.layer_norm2 = None
        self.output_layer = None

    def _data_generator(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        df = df.copy()
        df['order_id'] = pd.to_datetime(df['order_id'], utc=True)
        df = df[['user_id', 'skill_id', 'correct', 'order_id']]
        df = df.sort_values(by=['user_id', 'order_id'])
        user_ids = df['user_id'].unique()

        batch_past_exercises, batch_past_responses, batch_current_exercises, batch_targets = [], [], [], []

        for user_id in user_ids:
            user_data = df[df['user_id'] == user_id].sort_values('order_id')
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data['old_problem_id']]
            response_seq = user_data['correct'].astype(int).tolist()

            if len(exercise_seq) < 2:
                continue  # Skip users with less than 2 interactions

            for idx in range(1, len(exercise_seq)):
                start_idx = max(0, idx - self.num_steps)
                end_idx = idx
                past_exercises = exercise_seq[start_idx:end_idx]
                past_responses = response_seq[start_idx:end_idx]
                current_exercise = exercise_seq[idx]
                target_response = response_seq[idx]

                # Pad sequences if necessary
                if len(past_exercises) < self.num_steps:
                    pad_len = self.num_steps - len(past_exercises)
                    past_exercises = [0] * pad_len + past_exercises
                    past_responses = [0] * pad_len + past_responses

                batch_past_exercises.append(past_exercises)
                batch_past_responses.append(past_responses)
                batch_current_exercises.append(current_exercise)
                batch_targets.append(target_response)

                if len(batch_past_exercises) == self.batch_size:
                    yield (
                        np.array(batch_past_exercises, dtype=np.int32),
                        np.array(batch_past_responses, dtype=np.int32),
                        np.array(batch_current_exercises, dtype=np.int32),
                        np.array(batch_targets, dtype=np.float32)
                    )
                    batch_past_exercises, batch_past_responses, batch_current_exercises, batch_targets = [], [], [], []

        if batch_past_exercises:
            yield (
                np.array(batch_past_exercises, dtype=np.int32),
                np.array(batch_past_responses, dtype=np.int32),
                np.array(batch_current_exercises, dtype=np.int32),
                np.array(batch_targets, dtype=np.float32)
            )

    def _count_total_samples(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df['order_id'] = pd.to_datetime(df['order_id'], utc=True)
        df = df[['user_id', 'skill_id', 'correct', 'order_id']]
        df = df.sort_values(by=['user_id', 'order_id'])
        user_ids = df['user_id'].unique()

        total_samples = 0

        for user_id in user_ids:
            user_data = df[df['user_id'] == user_id].sort_values('order_id')
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data['skill_id']]

            if len(exercise_seq) < 2:
                continue  # Skip users with less than 2 interactions

            # Each user contributes (number of interactions - 1) samples
            total_samples += len(exercise_seq) - 1

        return total_samples

    @tf.function
    def _train_step(self, past_exercises, past_responses, current_exercises, y, optimizer, loss_fn) -> Tuple[tf.Tensor, tf.Tensor]:
      with tf.device('/GPU:0'):
          with tf.GradientTape() as tape:
              predictions = self([past_exercises, past_responses, current_exercises], training=True)
              loss = loss_fn(y, predictions)
          gradients = tape.gradient(loss, self.trainable_variables)
          optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      return loss, predictions


    @tf.function
    def _val_step(self, past_exercises, past_responses, current_exercises, y, loss_fn) -> Tuple[tf.Tensor, tf.Tensor]:
        predictions = self([past_exercises, past_responses, current_exercises], training=False)
        loss = loss_fn(y, predictions)
        return loss, predictions

    def preprocess(self, df: pd.DataFrame) -> None:
        df['order_id'] = pd.to_datetime(df['order_id'], utc=True)
        df = df[['user_id', 'problem_id', 'correct', 'order_id']]
        df = df.sort_values(by=['user_id', 'order_id'])
        unique_problems = df['problem_id'].unique()
        self.exercise_map = {problem: idx for idx, problem in enumerate(unique_problems, start=1)}
        self.num_exercises = len(self.exercise_map)
        self.exercise_embedding = tf.keras.layers.Embedding(input_dim=self.num_exercises + 1, output_dim=self.d_model)
        self.response_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=self.d_model)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_steps, output_dim=self.d_model)

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model, dropout=self.dropout_rate
        )
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
        past_exercises, past_responses, current_exercises = inputs
        attention_mask = tf.not_equal(past_exercises, 0)
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        past_exercise_embeddings = self.exercise_embedding(past_exercises)
        past_response_embeddings = self.response_embedding(past_responses)
        past_interactions_embeddings = past_exercise_embeddings + past_response_embeddings
        seq_len = tf.shape(past_exercises)[1]
        positions = tf.range(seq_len, dtype=tf.int32)
        position_embeddings = self.position_embedding(positions)
        position_embeddings = tf.expand_dims(position_embeddings, 0)
        past_interactions_embeddings += position_embeddings
        current_exercise_embedding = self.exercise_embedding(current_exercises)
        current_exercise_embedding = tf.expand_dims(current_exercise_embedding, axis=1)
        attention_output = self.multi_head_attention(
            query=current_exercise_embedding,
            key=past_interactions_embeddings,
            value=past_interactions_embeddings,
            attention_mask=attention_mask
        )
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layer_norm1(attention_output + current_exercise_embedding)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(ffn_output + out1)
        output = self.output_layer(out2)
        output = tf.squeeze(output, axis=[1, 2])
        return output

    def fit(self, train_df: pd.DataFrame, num_epochs: int = 10) -> None:
        self.preprocess(train_df)
        
        # Calculate total samples and iterations per epoch
        total_samples = self._count_total_samples(train_df)
        iterations_per_epoch = (total_samples + self.batch_size - 1) // self.batch_size

        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy()

        train_loss: tf.keras.metrics.Mean = tf.keras.metrics.Mean(name='train_loss')
        train_auc: tf.keras.metrics.AUC = tf.keras.metrics.AUC(name='train_auc')
        train_accuracy: tf.keras.metrics.BinaryAccuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        for epoch in range(num_epochs):
            train_loss.reset_state()
            train_auc.reset_state()
            train_accuracy.reset_state()

            train_data = self._data_generator(train_df)  # Reset the generator for each epoch

            train_pbar = tqdm.tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=100, total=iterations_per_epoch)
            for batch in train_pbar:
                past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
                loss, predictions = self._train_step(
                    past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch, optimizer, loss_fn
                )

                # Use tf.reshape to flatten tensors
                y_batch_flat = tf.reshape(y_batch, [-1])
                predictions_flat = tf.reshape(predictions, [-1])

                train_loss.update_state(loss)
                train_auc.update_state(y_batch_flat, predictions_flat)
                train_accuracy.update_state(y_batch_flat, predictions_flat)

                train_pbar.set_postfix({
                    'loss': f'{train_loss.result():.4f}',
                    'auc': f'{train_auc.result():.4f}',
                    'acc': f'{train_accuracy.result():.4f}'
                })

            print(f"Epoch {epoch + 1}/{num_epochs}: Loss={train_loss.result():.4f}, AUC={train_auc.result():.4f}, Accuracy={train_accuracy.result():.4f}")

    def eval(self, val_df: pd.DataFrame) -> Dict[str, float]:
        total_samples = self._count_total_samples(val_df)
        iterations = (total_samples + self.batch_size - 1) // self.batch_size

        val_loss: tf.keras.metrics.Mean = tf.keras.metrics.Mean(name='val_loss')
        val_auc: tf.keras.metrics.AUC = tf.keras.metrics.AUC(name='val_auc')
        val_accuracy: tf.keras.metrics.BinaryAccuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

        # Reset the states
        val_loss.reset_state()
        val_auc.reset_state()
        val_accuracy.reset_state()

        val_data = self._data_generator(val_df)

        val_pbar = tqdm.tqdm(val_data, desc=f"Evaluation [Val]", ncols=100, total=iterations)
        for batch in val_pbar:
            past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
            loss, predictions = self._val_step(
                past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch,
                tf.keras.losses.BinaryCrossentropy()
            )

            # Use tf.reshape to flatten tensors
            y_batch_flat = tf.reshape(y_batch, [-1])
            predictions_flat = tf.reshape(predictions, [-1])

            val_loss.update_state(loss)
            val_auc.update_state(y_batch_flat, predictions_flat)
            val_accuracy.update_state(y_batch_flat, predictions_flat)

            val_pbar.set_postfix({
                'loss': f'{val_loss.result():.4f}',
                'auc': f'{val_auc.result():.4f}',
                'acc': f'{val_accuracy.result():.4f}'
            })

        metrics: Dict[str, float] = {
            'loss': val_loss.result().numpy(),
            'auc': val_auc.result().numpy(),
            'accuracy': val_accuracy.result().numpy()
        }

        print(f"Validation: Loss={metrics['loss']:.4f}, AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return metrics
