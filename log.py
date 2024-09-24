import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Generator, Tuple
from tqdm import tqdm
import logging
import time
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    unique_skills = df['skill_id'].unique()
    unique_problems = df['old_problem_id'].unique()
    
    skill_map = {skill: idx for idx, skill in enumerate(unique_skills, start=1)}
    problem_map = {problem: idx for idx, problem in enumerate(unique_problems, start=1)}
    
    end_time = time.time()
    logging.info(f"Maps created. Unique skills: {len(skill_map)}, Unique problems: {len(problem_map)}")
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
    return skill_map, problem_map

def data_generator(df: pd.DataFrame, skill_map: dict, problem_map: dict, num_steps: int, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    user_ids = df['user_xid'].unique()
    num_users = len(user_ids)
    logging.info(f"Starting data generator. Total users: {num_users}")
    
    for start_idx in range(0, num_users, batch_size):
        end_idx = min(start_idx + batch_size, num_users)
        batch_user_ids = user_ids[start_idx:end_idx]
        logging.info(f"Processing batch of users {start_idx} to {end_idx}")
        
        batch_skills = []
        batch_problems = []
        batch_responses = []
        batch_targets = []
        
        for user_id in batch_user_ids:
            user_data = df[df['user_xid'] == user_id].sort_values('start_time')
            logging.debug(f"Processing user {user_id}. Interactions: {len(user_data)}")
            
            skill_seq = [skill_map[skill] for skill in user_data['skill_id']]
            problem_seq = [problem_map[problem] for problem in user_data['old_problem_id']]
            response_seq = user_data['discrete_score'].tolist()
            
            for i in range(1, len(skill_seq)):
                end = i
                start = max(0, end - num_steps)
                
                skills = skill_seq[start:end]
                problems = problem_seq[start:end]
                responses = response_seq[start:end]
                target = response_seq[end]
                
                if len(skills) < num_steps:
                    padding = num_steps - len(skills)
                    skills = [0] * padding + skills
                    problems = [0] * padding + problems
                    responses = [0] * padding + responses
                else:
                    skills = skills[-num_steps:]
                    problems = problems[-num_steps:]
                    responses = responses[-num_steps:]
                
                batch_skills.append(skills)
                batch_problems.append(problems)
                batch_responses.append(responses)
                batch_targets.append(target)
        
        logging.info(f"Yielding batch. Shape: ({len(batch_skills)}, {num_steps})")
        yield (np.array(batch_skills, dtype=np.int32),
               np.array(batch_problems, dtype=np.int32),
               np.array(batch_responses, dtype=np.int32),
               np.array(batch_targets, dtype=np.float32))

class SAKTModel(tf.keras.Model):
    def __init__(self, num_skills, num_problems, num_steps, d_model=256, num_heads=8, dropout_rate=0.2):
        super(SAKTModel, self).__init__()
        self.num_skills = num_skills
        self.num_problems = num_problems
        self.num_steps = num_steps
        self.d_model = d_model

        self.skill_embedding = tf.keras.layers.Embedding(input_dim=num_skills+1, output_dim=d_model)
        self.problem_embedding = tf.keras.layers.Embedding(input_dim=num_problems+1, output_dim=d_model)
        self.correctness_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=d_model)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=num_steps, output_dim=d_model)
        
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def create_attention_mask(self, batch_size):
        mask = tf.linalg.band_part(tf.ones((self.num_steps, self.num_steps)), -1, 0)
        mask = tf.reshape(mask, [1, 1, self.num_steps, self.num_steps])
        return tf.tile(mask, [batch_size, 1, 1, 1])

    def call(self, inputs, training=False):
        skills, problems, prev_responses = inputs
        
        batch_size = tf.shape(skills)[0]
        positions = tf.range(start=0, limit=self.num_steps, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [batch_size, 1])
        
        skill_embed = self.skill_embedding(skills)
        problem_embed = self.problem_embedding(problems)
        correctness_embed = self.correctness_embedding(prev_responses)
        pos_embed = self.pos_embedding(positions)
        
        x = skill_embed + problem_embed + correctness_embed + pos_embed
        
        attention_mask = self.create_attention_mask(batch_size)
        attention_mask = tf.cast(attention_mask, dtype=tf.bool)
        
        attn_output = self.multi_head_attention(query=x, key=x, value=x, attention_mask=attention_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        
        output = self.output_layer(out2)
        return output[:, -1, 0]  # Return only the last prediction for each sequence

@tf.function
def train_step(model, x_skills, x_problems, x_responses, y, optimizer, loss_fn, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model([x_skills, x_problems, x_responses], training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(y, predictions)
    return loss

@tf.function
def val_step(model, x_skills, x_problems, x_responses, y, loss_fn, val_accuracy):
    predictions = model([x_skills, x_problems, x_responses], training=False)
    loss = loss_fn(y, predictions)
    val_accuracy.update_state(y, predictions)
    return loss

def train_model(model, train_data, val_data, num_epochs=10, batch_size=64):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    val_accuracy = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        train_loss = 0
        train_accuracy.reset_state()
        train_start_time = time.time()
        for batch_num, batch in enumerate(tqdm(train_data, desc="Training")):
            x_skills_batch, x_problems_batch, x_responses_batch, y_batch = batch
            loss = train_step(model, x_skills_batch, x_problems_batch, x_responses_batch, y_batch, optimizer, loss_fn, train_accuracy)
            train_loss += loss.numpy()
            if batch_num % 100 == 0:
                logging.info(f"Batch {batch_num}, Loss: {loss.numpy():.4f}")
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        train_loss /= (batch_num + 1)
        logging.info(f"Epoch {epoch+1} Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy.result():.4f}, Time: {train_time:.2f}s")
        
        # Validation loop
        val_loss = 0
        val_accuracy.reset_states()
        val_start_time = time.time()
        for batch_num, batch in enumerate(tqdm(val_data, desc="Validation")):
            x_skills_batch, x_problems_batch, x_responses_batch, y_batch = batch
            loss = val_step(model, x_skills_batch, x_problems_batch, x_responses_batch, y_batch, loss_fn, val_accuracy)
            val_loss += loss.numpy()
            if batch_num % 100 == 0:
                logging.info(f"Validation Batch {batch_num}, Loss: {loss.numpy():.4f}")
        
        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        val_loss /= (batch_num + 1)
        logging.info(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy.result():.4f}, Time: {val_time:.2f}s")
def main():
    file_path = '23-24-problem_logs.csv'
    num_steps = 100
    batch_size = 64
    
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
    train_model(model, train_data, val_data)
    
    logging.info("Execution completed")

if __name__ == "__main__":
    main()