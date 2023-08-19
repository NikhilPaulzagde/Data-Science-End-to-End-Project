from flask import Flask
from src.logger import logging
from src.exception import CustomException


app=Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    logging.info("I m testing second way of logging")
    return "Welcome"


if __name__=="__main__":
    app.run(debug=True)