import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List
from tqdm import tqdm

def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Load the dataset from a CSV file, process the data by encoding problem IDs and skill IDs.
    
    Args:
        file_path (str): The path to the dataset file in CSV format.
    
    Returns:
        Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]: 
            - Processed DataFrame.
            - Encoder for problem IDs.
            - Encoder for skill IDs.
    """
    df: pd.DataFrame = pd.read_csv(file_path)
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
    df = df[['user_xid', 'old_problem_id', 'skill_id', 'discrete_score', 'start_time']]
    df = df.sort_values(by=['user_xid', 'start_time'])
    problem_encoder: LabelEncoder = LabelEncoder()
    skill_encoder: LabelEncoder = LabelEncoder()
    df['encoded_problem_id'] = problem_encoder.fit_transform(df['old_problem_id'])
    df['encoded_skill_id'] = skill_encoder.fit_transform(df['skill_id'])

    return df, problem_encoder, skill_encoder


def process_data_for_model(grouped_data: pd.DataFrame, num_steps: int) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Process grouped data for model training by padding sequences of problem IDs, skill IDs, and scores.
    
    Args:
        grouped_data (pd.DataFrame): Grouped data containing sequences of problem IDs, skill IDs, and scores.
        num_steps (int): The maximum sequence length for padding.
    
    Returns:
        List[Tuple[List[int], List[int], List[int]]]: 
            A list of tuples where each tuple contains:
            - Padded sequence of problem IDs.
            - Padded sequence of skill IDs.
            - Padded sequence of scores.
    """
    users_data: List[Tuple[List[int], List[int], List[int]]] = []
    for _, row in grouped_data.iterrows():
        sequence: List[Tuple[int, int, int]] = row['problem_skill_score']
        problem_ids: List[int] = [p[0] for p in sequence]
        skill_ids: List[int] = [p[1] for p in sequence]
        scores: List[int] = [p[2] for p in sequence]
        if len(problem_ids) > num_steps:
            problem_ids = problem_ids[:num_steps]
            skill_ids = skill_ids[:num_steps]
            scores = scores[:num_steps]
        else:
            padding: List[int] = [0] * (num_steps - len(problem_ids))
            problem_ids.extend(padding)
            skill_ids.extend(padding)
            scores.extend(padding)

        users_data.append((problem_ids, skill_ids, scores))
    
    return users_data
