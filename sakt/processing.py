import logging
import time
import os
from typing import Tuple, Generator
import pandas as pd
import numpy as np

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
                batch_responses.append(response_seq[:-1]) # all but first
                batch_targets.append(response_seq[1:]) # all but last
            
            yield (np.array(batch_skills, dtype=np.int32),
                   np.array(batch_problems, dtype=np.int32),
                   np.array(batch_responses, dtype=np.int32),
                   np.array(batch_targets, dtype=np.float32))

