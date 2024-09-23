from processing import load_and_process_data, process_data_for_model
from model import SAKTModel
from train import run_epoch
from viz import plot_attention, plot_latent_topics
import tensorflow as tf

def main():
    file_path = 'subset_23-24-problem_logs.csv' # CHANGE TO WORK W 5 FOLD CV
    num_steps = 50
    hidden_units = 200
    dropout_rate = 0.2
    num_heads = 8
    df, problem_encoder, skill_encoder = load_and_process_data(file_path)
    max_encoded_value = max(df['encoded_problem_id'].max(), df['encoded_skill_id'].max())
    num_skills = max_encoded_value + 1
    grouped_data = df.groupby('user_xid').apply(
        lambda x: list(zip(x['encoded_problem_id'], x['encoded_skill_id'], x['discrete_score']))
    ).reset_index(name='problem_skill_score')
    train_data = process_data_for_model(grouped_data, num_steps)
    model = SAKTModel(num_skills=num_skills, num_steps=num_steps, hidden_units=hidden_units, dropout_rate=dropout_rate, num_heads=num_heads)
    
    for epoch in range(1, 6):
        print(f"\nEpoch {epoch} / 5")
        rmse, auc = run_epoch(model, train_data, num_skills, num_steps, is_training=True)
        print(f'Epoch: {epoch}, Train RMSE: {rmse:.3f}, Train AUC: {auc:.3f}')
        if epoch == 1:
            inputs = (tf.convert_to_tensor(train_data[0][0]), tf.convert_to_tensor(train_data[0][1]), [0], [0])
            _, _, attention_weights = model.call(inputs, training=False)
            plot_attention(attention_weights.numpy())
        if epoch == 5:
            problem_embeddings = model.enc_embedding(tf.range(df['encoded_problem_id'].max() + 1)).numpy()
            skill_embeddings = model.enc_embedding(tf.range(df['encoded_skill_id'].max() + 1)).numpy()
            plot_latent_topics(problem_embeddings, skill_embeddings)

if __name__ == "__main__":
    main()
