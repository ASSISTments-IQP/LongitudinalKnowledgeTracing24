import tensorflow as tf
from typing import Tuple, Generator, Dict
import pandas as pd
import numpy as np
import tqdm


class SAKTModel(tf.keras.Model):
    def __init__(self, num_exercises: int = 50, num_steps: int = 100, batch_size: int = 32, d_model: int = 256,
                 num_heads: int = 8, dropout_rate: float = 0.2):
        super(SAKTModel, self).__init__()
        self.num_exercises = num_exercises
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.exercise_embedding = tf.keras.layers.Embedding(input_dim=num_exercises + 1, output_dim=d_model)
        self.response_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=d_model)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_steps, output_dim=d_model)

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
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

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df[['user_xid', 'old_problem_id', 'discrete_score', 'start_time']]
        df = df.sort_values(by=['user_xid', 'start_time'])
        unique_problems = df['old_problem_id'].unique()
        exercise_map = {problem: idx for idx, problem in enumerate(unique_problems, start=1)}
        df['old_problem_id'] = df['old_problem_id'].map(exercise_map)
        return df

    def _data_generator(self, df: pd.DataFrame) -> Generator[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        user_ids = df['user_xid'].unique()

        batch_past_exercises, batch_past_responses, batch_current_exercises, batch_targets = [], [], [], []

        for user_id in user_ids:
            user_data = df[df['user_xid'] == user_id].sort_values('start_time')
            exercise_seq = user_data['old_problem_id'].tolist()
            response_seq = user_data['discrete_score'].astype(int).tolist()

            if len(exercise_seq) < 2:
                continue

            for idx in range(1, len(exercise_seq)):
                start_idx = max(0, idx - self.num_steps)
                end_idx = idx
                past_exercises = exercise_seq[start_idx:end_idx]
                past_responses = response_seq[start_idx:end_idx]
                current_exercise = exercise_seq[idx]
                target_response = response_seq[idx]

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

    @tf.function
    def _train_step(self, past_exercises, past_responses, current_exercises, y, optimizer, loss_fn) -> Tuple[
        tf.Tensor, tf.Tensor]:
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

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, num_epochs: int = 10) -> None:
        train_df = self.preprocess(train_df)
        val_df = self.preprocess(val_df)

        train_data = self._data_generator(train_df)
        val_data = self._data_generator(val_df)

        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy()

        train_loss: tf.keras.metrics.Mean = tf.keras.metrics.Mean(name='train_loss')
        train_auc: tf.keras.metrics.AUC = tf.keras.metrics.AUC(name='train_auc')
        train_accuracy: tf.keras.metrics.BinaryAccuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        val_loss: tf.keras.metrics.Mean = tf.keras.metrics.Mean(name='val_loss')
        val_auc: tf.keras.metrics.AUC = tf.keras.metrics.AUC(name='val_auc')
        val_accuracy: tf.keras.metrics.BinaryAccuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

        for epoch in range(num_epochs):
            train_loss.reset_states()
            train_auc.reset_states()
            train_accuracy.reset_states()

            val_loss.reset_states()
            val_auc.reset_states()
            val_accuracy.reset_states()

            train_pbar = tqdm.tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=100)
            for batch in train_pbar:
                past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
                loss, predictions = self._train_step(
                    past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch, optimizer, loss_fn
                )

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

            val_pbar = tqdm.tqdm(val_data, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", ncols=100)
            for batch in val_pbar:
                past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
                loss, predictions = self._val_step(
                    past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch, loss_fn
                )

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

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {train_loss.result():.4f}, AUC: {train_auc.result():.4f}, Accuracy: {train_accuracy.result():.4f}")
            print(
                f"Val Loss: {val_loss.result():.4f}, AUC: {val_auc.result():.4f}, Accuracy: {val_accuracy.result():.4f}")

    def eval(self, test_df: pd.DataFrame) -> Dict[str, float]:
        test_df = self.preprocess(test_df)
        test_data = self._data_generator(test_df)

        test_loss: tf.keras.metrics.Mean = tf.keras.metrics.Mean(name='test_loss')
        test_auc: tf.keras.metrics.AUC = tf.keras.metrics.AUC(name='test_auc')
        test_accuracy: tf.keras.metrics.BinaryAccuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

        loss_fn: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy()

        test_pbar = tqdm.tqdm(test_data, desc="Evaluation [Test]", ncols=100)
        for batch in test_pbar:
            past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
            loss, predictions = self._val_step(
                past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch, loss_fn
            )

            y_batch_flat = tf.reshape(y_batch, [-1])
            predictions_flat = tf.reshape(predictions, [-1])

            test_loss.update_state(loss)
            test_auc.update_state(y_batch_flat, predictions_flat)
            test_accuracy.update_state(y_batch_flat, predictions_flat)

            test_pbar.set_postfix({
                'loss': f'{test_loss.result():.4f}',
                'auc': f'{test_auc.result():.4f}',
                'acc': f'{test_accuracy.result():.4f}'
            })

        metrics: Dict[str, float] = {
            'loss': test_loss.result().numpy(),
            'auc': test_auc.result().numpy(),
            'accuracy': test_accuracy.result().numpy()
        }

        print(f"Test: Loss={metrics['loss']:.4f}, AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return metrics