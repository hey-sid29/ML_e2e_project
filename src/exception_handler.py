'''This file contains the exception handler to handle the exceptions raised in the project'''
import sys
import logging
from src.logger import logging
def get_error_details(error, error_details:sys):
    _,_, exc_tb = error_details.exc_info() # get the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename # get the file name
    line_number = exc_tb.tb_lineno # get the line number
    error_message = "Error occured in python script: {0} at line number: {1} with error: {2}".format(file_name, line_number, str(error))
    return error_message




class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message, error_details)

        def __str__(self):
            return self.error_message
        

