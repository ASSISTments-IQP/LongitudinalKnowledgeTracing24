from model import SAKTModel, train_step, val_step
from typing import Generator
import tensorflow as tf
import logging
import time
from datetime import datetime
import tqdm



def train_model(model: SAKTModel, train_data: Generator, val_data: Generator, num_epochs: int = 10, batch_size: int = 32, steps_per_epoch: int = 1000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_auc = tf.keras.metrics.AUC(name='train_auc')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_auc = tf.keras.metrics.AUC(name='val_auc')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    logging.info("Starting model training")
    logging.info(f"Model configuration: batch_size={batch_size}, steps_per_epoch={steps_per_epoch}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}, Learning rate: {optimizer.learning_rate.numpy()}")

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        train_loss.reset_state()
        train_auc.reset_state()
        train_accuracy.reset_state()
        train_start_time = time.time()
        
        train_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs} [Train]", ncols=100)
        for batch_num in train_pbar:
            batch = next(train_data)
            x_skills_batch, x_problems_batch, x_responses_batch, y_batch = batch
            loss, predictions = train_step(model, x_skills_batch, x_problems_batch, x_responses_batch, y_batch, optimizer, loss_fn)
            
            train_loss.update_state(loss)
            train_auc.update_state(y_batch, predictions)
            train_accuracy.update_state(y_batch, predictions)
            
            train_pbar.set_postfix({
                'loss': f'{train_loss.result():.4f}',
                'auc': f'{train_auc.result():.4f}',
                'acc': f'{train_accuracy.result():.4f}'
            })
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        logging.info(f"Epoch {epoch+1} Training - Loss: {train_loss.result():.4f}, AUC: {train_auc.result():.4f}, "
                     f"Accuracy: {train_accuracy.result():.4f}, Time: {train_time:.2f}s")
        val_loss.reset_state()
        val_auc.reset_state()
        val_accuracy.reset_state()
        val_start_time = time.time()
        
        val_pbar = tqdm(range(steps_per_epoch // 10), desc=f"Epoch {epoch+1}/{num_epochs} [Val]", ncols=100)
        for batch_num in val_pbar:
            batch = next(val_data)
            x_skills_batch, x_problems_batch, x_responses_batch, y_batch = batch
            loss, predictions = val_step(model, x_skills_batch, x_problems_batch, x_responses_batch, y_batch, loss_fn)
            
            val_loss.update_state(loss)
            val_auc.update_state(y_batch, predictions)
            val_accuracy.update_state(y_batch, predictions)
            
            val_pbar.set_postfix({
                'loss': f'{val_loss.result():.4f}',
                'auc': f'{val_auc.result():.4f}',
                'acc': f'{val_accuracy.result():.4f}'
            })
        
        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        logging.info(f"Epoch {epoch+1} Validation - Loss: {val_loss.result():.4f}, AUC: {val_auc.result():.4f}, "
                     f"Accuracy: {val_accuracy.result():.4f}, Time: {val_time:.2f}s")

    logging.info("Training completed")
