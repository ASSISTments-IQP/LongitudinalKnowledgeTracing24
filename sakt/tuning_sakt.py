import optuna
from SAKT_model import SAKTModel
import pandas as pd

train_df = pd.read_csv('../Data/samples/23-24/sample3.csv')
val_df = pd.read_csv('../Data/samples/23-24/sample4.csv')
def objective(trial):
    num_steps = trial.suggest_int('num_steps', 20, 100, step = 10),
    batch_size = trial.suggest_categorical('batch_size', 16, 64, step = 16),
    d_model = trial.suggest_int('d_model', 128, 512, step = 32),
    num_heads = trial.suggest_int('num_heads', 2, 16, step = 2),
    dropout_rate = trial.suggest_int('dropout_rate', 0.1, 0.5)
    
    model = SAKTModel(
        num_steps = num_steps,
        batch_size = batch_size,
        d_model = d_model,
        num_heads = num_heads,
        dropout_rate = dropout_rate
    )
    
    model.preprocess(train_df)
    model.fit(train_df, val_df, num_epochs = 5, early_stopping = True, patience = 2)
    val_auc = model.eval(val_df)['auc']
    return val_auc
    
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)


print("Best hyperparameters:", study.best_params)
print("Best validation AUC:", study.best_value)
    
    