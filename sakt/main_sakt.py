from model import SAKTModel
from train import train_model
from processing import load_and_process_data, create_skill_problem_maps, data_generator
import logging
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16') # we do a little memory efficiency

def main():
    file_path = 'sakt/subset.csv'
    num_steps = 50
    batch_size = 32
    num_epochs = 10
    steps_per_epoch = 10 # idk what to use here, or whether to use it at all actually
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
    train_model(model, train_data, val_data, num_epochs=num_epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
    
    logging.info("Saving model")
    model.save('sakt_model')
    
    logging.info("Execution completed")

if __name__ == "__main__":
    main()