import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Generator, Tuple, List
from tqdm import tqdm
import logging
import time
import os
from datetime import datetime

log_dir = "sakt/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])


