import logging
from datetime import datetime
from time import sleep
import random

run_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

log_file_path = f'logs/{run_start_time}_log.txt'

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(message)s',
#     handlers=[
#         logging.StreamHandler(),                   # To print to the terminal
#         logging.FileHandler(log_file_path),        # To save to the log file
#     ],
# )

# Create a custom formatter without the level name
formatter = logging.Formatter('%(message)s')

# Configure the root logger to use the custom formatter
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create a FileHandler for the log file with the custom formatter
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Create a StreamHandler for the terminal with the custom formatter
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)


for i in range(10):
    logging.info(f'Iteration {i}')
    logging.info("This is an informational message.")
    logging.info("================================")

    sleep(random.uniform(1, 10))