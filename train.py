import joblib
import sklearn.feature_extraction.text
from sklearn.linear_model import LogisticRegression
import pandas as pd
from omegaconf import OmegaConf
import sklearn.model_selection

def train(config):
    train_inputs = joblib.load(config.features.train_features_save_path)
    train_outputs = pd.read_csv(config.data.train_csv_file_path)['label']
    print(1111, train_outputs)
    penalty = config.train.penalty
    C = config.train.C
    solver = config.train.solver
    model = LogisticRegression(penalty=penalty, C=C, solver=solver)
    model.fit(train_inputs, train_outputs)

    joblib.dump(model, config.train.model_save_path)

if __name__ == '__main__':
    config = OmegaConf.load('params.yaml')
    train(config)
