import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Generator, Tuple, List
from tqdm import tqdm
import logging
import time
import os
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_and_process_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {file_path}")
    start_time = time.time()
    df = pd.read_csv(file_path)
    logging.info(f"Data loaded. Shape: {df.shape}")
    
    logging.info("Processing data")
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
    df = df[['user_xid', 'old_problem_id', 'skill_id', 'discrete_score', 'start_time']]
    df = df.sort_values(by=['user_xid', 'start_time'])
    
    end_time = time.time()
    logging.info(f"Data processing completed. Time taken: {end_time - start_time:.2f} seconds")
    return df

def create_skill_problem_maps(df: pd.DataFrame) -> Tuple[dict, dict]:
    logging.info("Creating skill and problem maps")
    start_time = time.time()
    
    skill_map = {skill: idx for idx, skill in enumerate(df['skill_id'].unique(), start=1)}
    problem_map = {problem: idx for idx, problem in enumerate(df['old_problem_id'].unique(), start=1)}
    
    end_time = time.time()
    logging.info(f"Maps created. Unique skills: {len(skill_map)}, Unique problems: {len(problem_map)}")
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
    return skill_map, problem_map

def data_generator(df: pd.DataFrame, skill_map: dict, problem_map: dict, num_steps: int, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    user_ids = df['user_xid'].unique()
    num_users = len(user_ids)
    
    while True:
        np.random.shuffle(user_ids)
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)
            batch_user_ids = user_ids[start_idx:end_idx]
            
            batch_skills = []
            batch_problems = []
            batch_responses = []
            batch_targets = []
            
            for user_id in batch_user_ids:
                user_data = df[df['user_xid'] == user_id].sort_values('start_time')
                
                skill_seq = [skill_map[skill] for skill in user_data['skill_id']]
                problem_seq = [problem_map[problem] for problem in user_data['old_problem_id']]
                response_seq = user_data['discrete_score'].tolist()
                
                if len(skill_seq) > num_steps:
                    start_index = np.random.randint(0, len(skill_seq) - num_steps)
                    skill_seq = skill_seq[start_index:start_index + num_steps]
                    problem_seq = problem_seq[start_index:start_index + num_steps]
                    response_seq = response_seq[start_index:start_index + num_steps]
                else:
                    padding = num_steps - len(skill_seq)
                    skill_seq = [0] * padding + skill_seq
                    problem_seq = [0] * padding + problem_seq
                    response_seq = [0] * padding + response_seq
                
                batch_skills.append(skill_seq)
                batch_problems.append(problem_seq)
                batch_responses.append(response_seq[:-1])  # Use all but the last response as input
                batch_targets.append(response_seq[1:])  # Use all but the first response as target
            
            yield (np.array(batch_skills, dtype=np.int32),
                   np.array(batch_problems, dtype=np.int32),
                   np.array(batch_responses, dtype=np.int32),
                   np.array(batch_targets, dtype=np.float32))

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

def train_model(model: SAKTModel, train_data: Generator, val_data: Generator, num_epochs: int = 10, batch_size: int = 32):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_auc = tf.keras.metrics.AUC(name='train_auc')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_auc = tf.keras.metrics.AUC(name='val_auc')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    logging.info("Starting model training")
    logging.info(f"Model configuration: batch_size={batch_size}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}, Learning rate: {optimizer.learning_rate.numpy()}")

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        train_loss.reset_state()
        train_auc.reset_state()
        train_accuracy.reset_state()
        train_start_time = time.time()
        
        for batch_num, batch in enumerate(train_data):
            x_skills_batch, x_problems_batch, x_responses_batch, y_batch = batch
            loss, predictions = train_step(model, x_skills_batch, x_problems_batch, x_responses_batch, y_batch, optimizer, loss_fn)
            
            train_loss.update_state(loss)
            train_auc.update_state(y_batch, predictions)
            train_accuracy.update_state(y_batch, predictions)
            
            if (batch_num + 1) % 10 == 0:  # Log every 10 batches
                logging.info(f"Epoch {epoch+1}, Batch {batch_num+1}: "
                             f"Loss: {train_loss.result():.4f}, "
                             f"AUC: {train_auc.result():.4f}, "
                             f"Accuracy: {train_accuracy.result():.4f}")
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        logging.info(f"Epoch {epoch+1} Training - Loss: {train_loss.result():.4f}, AUC: {train_auc.result():.4f}, "
                     f"Accuracy: {train_accuracy.result():.4f}, Time: {train_time:.2f}s")
        
        # Validation loop
        val_loss.reset_states()
        val_auc.reset_states()
        val_accuracy.reset_states()
        val_start_time = time.time()
        
        for batch_num, batch in enumerate(val_data):
            x_skills_batch, x_problems_batch, x_responses_batch, y_batch = batch
            loss, predictions = val_step(model, x_skills_batch, x_problems_batch, x_responses_batch, y_batch, loss_fn)
            
            val_loss.update_state(loss)
            val_auc.update_state(y_batch, predictions)
            val_accuracy.update_state(y_batch, predictions)
            
            if (batch_num + 1) % 10 == 0:  # Log every 10 batches
                logging.info(f"Validation - Epoch {epoch+1}, Batch {batch_num+1}: "
                             f"Loss: {val_loss.result():.4f}, "
                             f"AUC: {val_auc.result():.4f}, "
                             f"Accuracy: {val_accuracy.result():.4f}")
        
        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        logging.info(f"Epoch {epoch+1} Validation - Loss: {val_loss.result():.4f}, AUC: {val_auc.result():.4f}, "
                     f"Accuracy: {val_accuracy.result():.4f}, Time: {val_time:.2f}s")

    logging.info("Training completed")

def main():
    file_path = '23-24-problem_logs.csv'
    num_steps = 50
    batch_size = 32
    num_epochs = 10
    
    logging.info("Starting main execution")
    
    df = load_and_process_data(file_path)
    
    skill_map, problem_map = create_skill_problem_maps(df)
    num_skills = len(skill_map)
    num_problems = len(problem_map)
    
    logging.info("Creating data generators")
    train_data = data_generator(df, skill_map, problem_map, num_steps, batch_size)
    val_data = data_generator(df, skill_map, problem_map, num_steps, batch_size)
    
    logging.info("Creating model")
    model = SAKTModel(num_skills=num_skills, num_problems=num_problems, num_steps=num_steps)
    
    logging.info("Starting model training")
    train_model(model, train_data, val_data, num_epochs=num_epochs, batch_size=batch_size)
    
    logging.info("Saving model")
    model.save('sakt_model')
    
    logging.info("Execution completed")

if __name__ == "__main__":
    main()