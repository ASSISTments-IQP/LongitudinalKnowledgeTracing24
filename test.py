import tensorflow as tf
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


df = pd.read_csv('samples/19-20/sample8.csv')
df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
df = df[['user_xid', 'old_problem_id', 'skill_id', 'discrete_score', 'start_time']]
df = df.sort_values(by=['user_xid', 'start_time'])
problem_encoder = LabelEncoder()
skill_encoder = LabelEncoder()

df['encoded_problem_id'] = problem_encoder.fit_transform(df['old_problem_id'])
df['encoded_skill_id'] = skill_encoder.fit_transform(df['skill_id'])

max_encoded_value = max(df['encoded_problem_id'].max(), df['encoded_skill_id'].max())
grouped_data = df.groupby('user_xid').apply(
    lambda x: list(zip(x['encoded_problem_id'], x['encoded_skill_id'], x['discrete_score']))
).reset_index(name='problem_skill_score')

def process_data_for_model(grouped_data, num_steps):
    users_data = []
    for _, row in grouped_data.iterrows():
        sequence = row['problem_skill_score']
        problem_ids = [p[0] for p in sequence]
        skill_ids = [p[1] for p in sequence]
        scores = [p[2] for p in sequence]

        if len(problem_ids) > num_steps:
            problem_ids = problem_ids[:num_steps]
            skill_ids = skill_ids[:num_steps]
            scores = scores[:num_steps]
        else:
            padding = [0] * (num_steps - len(problem_ids))
            problem_ids.extend(padding)
            skill_ids.extend(padding)
            scores.extend(padding)

        users_data.append((problem_ids, skill_ids, scores))
    
    return users_data

num_steps = 200
num_skills = max_encoded_value + 1
hidden_units = 200
dropout_rate = 0.2
num_heads = 8
train_data = process_data_for_model(grouped_data, num_steps)


class Model(tf.keras.Model):
    def __init__(self, num_skills=50, num_steps=200, hidden_units=200, dropout_rate=0.2, num_heads=2):
        super(Model, self).__init__()
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



    def call(self, inputs, training=False):
        x, problems, target_id, target_correctness = inputs
        x = tf.cast(x, tf.int32)
        key_masks = tf.cast(tf.not_equal(x, 0), tf.float32)
        enc = self.enc_embedding(x) + self.pos_embedding(tf.range(self.num_steps - 1))
        enc = enc * tf.expand_dims(key_masks, axis=-1)
        enc = self.dropout(enc, training=training)
        attn_output = self.multihead_attention(enc, enc, training=training)
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
        return pred, loss

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            pred, loss = self.call(inputs, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
def run_epoch(model, data, num_skills, num_steps, batch_size=128, is_training=True):
    actual_labels = []
    pred_labels = []
    index = 0
    with tqdm(total=len(data) // batch_size) as pbar:
        while index + batch_size < len(data):
            x = np.zeros((batch_size, num_steps - 1))
            problems = np.zeros((batch_size, num_steps - 1))
            target_id = []
            target_correctness = []

            for i in range(batch_size):
                problem_ids, skill_ids, correctness = data[index + i]
                for j in range(num_steps - 1):
                    problem_id = int(problem_ids[j])
                    label_index = problem_id + (num_skills if int(correctness[j]) else 0)
                    x[i, j] = label_index
                    problems[i, j] = problem_ids[j + 1] if j + 1 < num_steps else 0
                    target_id.append(i * (num_steps - 1) + j)
                    target_correctness.append(correctness[j + 1] if j + 1 < num_steps else 0)
                    actual_labels.append(correctness[j + 1] if j + 1 < num_steps else 0)

            index += batch_size

            inputs = (x, problems, target_id, target_correctness)
            pred, _ = model.call(inputs, training=is_training)

            pred_labels.extend(pred.numpy())

            pbar.update(1)

    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, _ = metrics.roc_curve(actual_labels, pred_labels)
    auc = metrics.auc(fpr, tpr)
    return rmse, auc

def main():
    train_data = process_data_for_model(grouped_data, num_steps)
    model = Model(num_skills=num_skills, num_steps=num_steps, hidden_units=hidden_units, dropout_rate=dropout_rate, num_heads=num_heads)
    for epoch in range(1, 6):
        print(f"\nEpoch {epoch} / 5")
        rmse, auc = run_epoch(model, train_data, num_skills, num_steps, is_training=True)
        print(f'Epoch: {epoch}, Train RMSE: {rmse:.3f}, Train AUC: {auc:.3f}')

        if epoch % 5 == 0:
            rmse, auc = run_epoch(model, train_data, num_skills, num_steps, is_training=False)
            print(f'Epoch: {epoch}, Test RMSE: {rmse:.3f}, Test AUC: {auc:.3f}')


main()
