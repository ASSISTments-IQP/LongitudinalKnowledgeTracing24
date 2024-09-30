import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Data sequencing (sorting)
df = pd.read_csv("../../23-24-problem_logs.csv")
df = df[['user_xid', 'old_problem_id', 'skill_id', 'discrete_score', 'start_time']]
df = df.sort_values(by=['user_xid', 'old_problem_id', 'start_time'])
gk = df.groupby(by=['user_xid', 'skill_id'])['discrete_score'].apply(list)
print(gk.head(100))


# Model
num_states = 2                # two states "mastered" and "unmastered"
updated_model_params = "e"    # all parameters are initialized explicitly
model = CategoricalHMM(n_components=num_states, init_params=updated_model_params, n_iter=1000)

start_prob = np.array([0.5, 0.5])
trans_prob = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
])
emission_prob = np.array([
    [0.8, 0.2],
    [0.2, 0.8]
])

model.startprob_ = start_prob
model.transmat_ = trans_prob
model.emission_prob_ = emission_prob

x = np.concatenate(gk.values).reshape(-1, 1) # reshape into array
lens = [len(seq) for seq in gk]              # the length of each sequence in our grouped dataframe
print('Begin training.')
model.fit(x, lens)

trained_start_prob = model.startprob_
trained_trans_prob = model.transmat_
trained_emission_prob = model.emissionprob_

print('Trained Start_Probabilities: ' , trained_start_prob)
print('Trained Transition Probabilities: ', trained_trans_prob)
print('Trained Emission Probabilities: ', trained_emission_prob)
print('Finished training.')

print(model.monitor_)
print(model.monitor_.converged)

predicted_states = []
for seq in gk:
    seq = np.array(seq).reshape(-1, 1)
    predicted_states.append(model.predict(seq))
predicted_states_flat = np.concatenate(predicted_states)
state_to_correct = model.emissionprob_[:, 1] > 0.5
predicted_performance = state_to_correct[predicted_states_flat]
actual_scores = np.concatenate(gk.values)
accuracy = accuracy_score(actual_scores, predicted_performance)
f1 = f1_score(actual_scores, predicted_performance)
rmse = np.sqrt(mean_squared_error(actual_scores, predicted_performance))
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print('RMSE:', rmse)


