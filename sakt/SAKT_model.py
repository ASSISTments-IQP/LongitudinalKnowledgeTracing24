import tensorflow as tf
from typing import Tuple, Generator, Dict
import pandas as pd
import numpy as np
import tqdm

class SAKTModel(tf.keras.Model):
    def __init__(self, num_steps: int = 50, batch_size: int = 16, d_model: int = 128,
                 num_heads: int = 8, dropout_rate: float = 0.2, init_learning_rate = 1e-3, verbose=1, gpu_num=0, problem=True):
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
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

        if problem:
            self.vocab_col = 'old_problem_id'
        else:
            self.vocab_col = 'skill_id'

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.device = f"/GPU:{gpu_num}" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        print(f"Using device: {self.device}")
    
            
    @tf.function
    def _train_step(self, past_exercises, past_responses, current_exercises, y):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                predictions = self([past_exercises, past_responses, current_exercises], training=True)
                loss = self.loss_fn(y, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, predictions

    @tf.function
    def _val_step(self, past_exercises, past_responses, current_exercises, y):
        with tf.device(self.device):
            predictions = self([past_exercises, past_responses, current_exercises], training=False)
            loss = self.loss_fn(y, predictions)
        return loss, predictions

    
    

    def _data_generator(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        df = df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df[['user_xid', self.vocab_col, 'discrete_score', 'start_time']].sort_values(by=['user_xid', 'start_time'])
        user_xids = df['user_xid'].unique()

        batch_past_exercises, batch_past_responses, batch_current_exercises, batch_targets = [], [], [], []

        for user_xid in user_xids:
            user_data = df[df['user_xid'] == user_xid].sort_values('start_time')
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data[self.vocab_col]]
            response_seq = user_data['discrete_score'].astype(int).tolist()


            for idx in range(1, len(exercise_seq)):
                start_idx = max(0, idx - self.num_steps)
                past_exercises = exercise_seq[start_idx:idx]
                past_responses = response_seq[start_idx:idx]
                current_exercise = exercise_seq[idx]
                target_response = response_seq[idx]
                
                if len(past_exercises) > self.num_steps:
                    past_exercises = past_exercises[-self.num_steps:]
                    past_responses = past_responses[-self.num_steps:]

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
            pad_size = self.batch_size - len(batch_past_exercises)
            if pad_size > 0:
                pad_exercises = [ [0]*self.num_steps ] * pad_size
                pad_responses = [ [0]*self.num_steps ] * pad_size
                pad_current_exercises = [0] * pad_size
                pad_targets = [0.0] * pad_size

                batch_past_exercises.extend(pad_exercises)
                batch_past_responses.extend(pad_responses)
                batch_current_exercises.extend(pad_current_exercises)
                batch_targets.extend(pad_targets)

            yield (
                np.array(batch_past_exercises, dtype=np.int32),
                np.array(batch_past_responses, dtype=np.int32),
                np.array(batch_current_exercises, dtype=np.int32),
                np.array(batch_targets, dtype=np.float32)
            )

    def _count_total_samples(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df[['user_xid', self.vocab_col, 'discrete_score', 'start_time']].sort_values(by=['user_xid', 'start_time'])
        user_xids = df['user_xid'].unique()

        total_samples = 0
        for user_xid in user_xids:
            user_data = df[df['user_xid'] == user_xid]
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data[self.vocab_col]]
            if len(exercise_seq) >= 2:
                total_samples += len(exercise_seq) - 1

        return total_samples


    def preprocess(self, df: pd.DataFrame) -> None:
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df[['user_xid', self.vocab_col, 'discrete_score', 'start_time']].sort_values(by=['user_xid', 'start_time'])
        unique_problems = df[self.vocab_col].unique()
        self.exercise_map = {problem: idx for idx, problem in enumerate(unique_problems, start=1)}
        self.num_exercises = len(self.exercise_map)
        self.exercise_embedding = tf.keras.layers.Embedding(input_dim=self.num_exercises + 1, output_dim=self.d_model)
        self.response_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=self.d_model)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_steps, output_dim=self.d_model)
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model, dropout=self.dropout_rate)
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
        past_exercise_embeddings = self.exercise_embedding(past_exercises)
        past_response_embeddings = self.response_embedding(past_responses)
        past_interactions_embeddings = past_exercise_embeddings + past_response_embeddings

        seq_len = tf.shape(past_exercises)[1]
        positions = tf.range(seq_len, dtype=tf.int32)
        position_embeddings = self.position_embedding(positions)
        past_interactions_embeddings += tf.expand_dims(position_embeddings, 0)

        current_exercise_embedding = self.exercise_embedding(current_exercises)
        attention_output = self.multi_head_attention(
            query=tf.expand_dims(current_exercise_embedding, axis=1),
            key=past_interactions_embeddings,
            value=past_interactions_embeddings,
            attention_mask=tf.cast(attention_mask, dtype=tf.int32)
        )
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layer_norm1(attention_output + tf.expand_dims(current_exercise_embedding, axis=1))

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(ffn_output + out1)

        output = self.output_layer(out2)
        return tf.squeeze(output[:, -1, :], axis = 1)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, num_epochs: int = 4, early_stopping: bool = True, patience: int = 1):
        self.preprocess(train_df)
        total_samples = self._count_total_samples(train_df)
        iterations_per_epoch = (total_samples + self.batch_size - 1) // self.batch_size
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_auc = tf.keras.metrics.AUC(name='train_auc')
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        best_val_auc = 0.0
        epochs_since_improvement = 0


        for epoch in range(num_epochs):
            train_loss.reset_state()
            train_auc.reset_state()
            train_accuracy.reset_state()

            train_data = self._data_generator(train_df)
            train_pbar = tqdm.tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=100, total=iterations_per_epoch)
            for batch in train_pbar:
                past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
                loss, predictions = self._train_step(past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch)

                train_loss.update_state(loss)
                train_auc.update_state(y_batch, predictions)
                train_accuracy.update_state(y_batch, predictions)

                train_pbar.set_postfix({
                    'loss': f'{train_loss.result():.4f}',
                    'auc': f'{train_auc.result():.4f}',
                    'acc': f'{train_accuracy.result():.4f}'
                })

            print(f"Epoch {epoch + 1}/{num_epochs}: Loss={train_loss.result():.4f}, AUC={train_auc.result():.4f}, Accuracy={train_accuracy.result():.4f}")

            if val_df is not None:
                val_metrics = self.eval(val_df)
                val_auc_score = val_metrics['auc']

                if val_auc_score > best_val_auc:
                    best_val_auc = val_auc_score
                    epochs_since_improvement = 0
                    self.save_weights('best_model_weights.h5')
                else:
                    epochs_since_improvement += 1

                if early_stopping and epochs_since_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        if val_df is not None and early_stopping:
            self.load_weights('best_model_weights.h5')

    def eval(self, val_df: pd.DataFrame) -> Dict[str, float]:
        total_samples = self._count_total_samples(val_df)
        iterations = (total_samples + self.batch_size - 1) // self.batch_size

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_auc = tf.keras.metrics.AUC(name='val_auc')
        val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

        val_loss.reset_state()
        val_auc.reset_state()
        val_accuracy.reset_state()

        val_data = self._data_generator(val_df)

        val_pbar = tqdm.tqdm(val_data, desc="Evaluation [Val]", ncols=100, total=iterations)
        for batch in val_pbar:
            past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch = batch
            loss, predictions = self._val_step(past_exercises_batch, past_responses_batch, current_exercises_batch, y_batch)
            
            val_loss.update_state(loss)
            val_auc.update_state(y_batch, predictions)
            val_accuracy.update_state(y_batch, predictions)

            val_pbar.set_postfix({
                'loss': f'{val_loss.result():.4f}',
                'auc': f'{val_auc.result():.4f}',
                'acc': f'{val_accuracy.result():.4f}'
            })

        metrics = {
            'loss': val_loss.result().numpy(),
            'auc': val_auc.result(),
            'accuracy': val_accuracy.result().numpy()
        }

        print(f"Validation: Loss={metrics['loss']:.4f}, AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return float(metrics['auc'])
 


