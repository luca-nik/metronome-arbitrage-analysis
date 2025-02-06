import logging
import os

# Configure the logger
logging.basicConfig(
    filename="data/strategy.log",
    filemode="w",  # Overwrite the log file each run
    level=logging.INFO,
    format="%(asctime)s - %(message)s",  # Include timestamps if desired
)

# Provide a function to log messages
def log_message(message: str):
    """
    Logs a message to the strategy.log file.

    Args:
        message (str): The message to log.
    """
    logging.info(message)

def prettify_time(time: float, include_milliseconds: bool = False) -> str:
    """
    Converts a time in seconds into a string of the format 'xx:yy:zz', 
    where xx is hours, yy is minutes, and zz is seconds.
    
    Args:
        time (float): Time in seconds.
    
    Returns:
        str: Time formatted as 'xx:yy:zz' where xx is hours, yy is minutes, and zz is seconds.
    """
    # Calculate hours, minutes, and seconds
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)

    # Return the appropriate format
    if include_milliseconds:
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    else:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    