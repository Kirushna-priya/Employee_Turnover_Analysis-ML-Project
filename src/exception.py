import sys
import traceback
import logging
from src.logger import logging



def error_message_detail(error_message, error_detail: sys):
    _, _, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{error_message}]"
    else:
        # Fallback when no traceback exists
        return f"Error message: {error_message}"
    

class CustomException(Exception):
    def __init__(self,error_message,error_detail=None):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message
    
