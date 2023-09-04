import  os,sys
from src.logger import logging

#exc_tb-> execution from try block --> it helps to run code from try block's first line 
#exc.info --> function that gives full information of code that we execute
#tb_frame --> Try block frame --> it helps exc_tb to run the code line by line and find the error 



def error_message_detailed(error,error_detailed:sys):
    _,_,exc_tb=error_detailed.exc_info()

    file_name=exc_tb.tb_frame.f_code.co_filename


#this variable is made to show the error message that comes along with error
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno, str(error)
    )

    return error_message
'''In short, this Python code defines a function `error_message_detailed` that takes two arguments: `error` and `error_detailed`. 

        Inside the function:
        1. It extracts information from the `error_detailed` object to identify the Python script file name and line number where an error occurred.
        2. It constructs an error message that includes the script name, line number, and the provided `error` message.
        3. Finally, it returns this constructed error message.

        This function is designed to provide a more detailed error message by including information about where and why an error occurred in a Python script.'''


# Making a custom class for error message   to collect the error message
class CustomException(Exception):
    def __init__(self,error_message ,  error_detailed: sys):
        super().__init__(error_message)
        self.error_message = error_message_detailed(error_message,error_detailed=error_detailed)

    def __str__(self) :
        return self.error_message
    
'''This code defines a custom exception class `CustomException` that combines a user-provided error message with additional details from an `error_detailed` object. When raised, this exception displays the detailed error message when converted to a string.'''



