import os
import sys
import logging
from datetime import datetime

#Logging format  defining
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#Making Path for log files using getcwd
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(log_path , exist_ok=True)

#joining Log_FILE and log_path for filename
Log_file_path=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=Log_file_path,
    format= '[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


'''In short, this code sets up logging in Python to:

1. Write log messages to a specified log file.
2. Format log messages to include a timestamp, line number, logger name, log level, and message.
3. Record log messages with a severity level of `INFO` and higher in the log file.'''


