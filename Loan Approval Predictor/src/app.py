from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
import numpy as np

app = Flask(__name__)

@app.route("/",methods = ["GET", "POST"])
def prediction_data():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        input_data = CustomClass(
            Gender= int(request.form.get("gender")),
            Married =int(request.form.get("married")),
            Dependents=int(request.form.get("dependents")),
            Education=int(request.form.get("education")),
            Self_Employed=int(request.form.get("self_employed")),
            ApplicantIncome=int(request.form.get("applicant_income")),
            CoapplicantIncome=int(request.form.get("coapplicant_income")),
            LoanAmount=int(request.form.get("loan_amount")),
            Loan_Amount_Term=int(request.form.get("loan_amount_term")),
            Credit_History=int(request.form.get("credit_history")),
            Property_Area=int(request.form.get("property_area")),
        )

        final_data =input_data.DataFrame()
       
       
        pipeline_predict=PredictionPipeline()
        pred=pipeline_predict.predict(final_data)

        result=pred

        if result==0:
                return render_template("result.html", final_result = "Your Are Not Eligible:{}".format(result) )


        else:
             return render_template("result.html", final_result = "Your Are  Eligible:{}".format(result) )



       

if __name__ == "__main__":
     app.run(host = "0.0.0.0", debug = True)

