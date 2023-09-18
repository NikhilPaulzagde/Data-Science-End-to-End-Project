from flask import Flask,request, render_template
from src.pipeline.prediction_pipeline import CustomClass,PredictionPipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def prediction_data():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        data = CustomClass (
            cement =request.form.get('cement'),
            blast_furnace_slag = request.form.get('blast_furnace_slag'),
            fly_ash = request.form.get('fly_ash'),
            water = request.form.get('water'),
            superplasticizer = request.form.get('superplasticizer'),
            coarse_aggregate = request.form.get('coarse_aggregate'),
            fine_aggregate = request.form.get('fine_aggregate'),
            age = request.form.get('age')
        )

        pred_df = data.get_Dataframe()
        print(pred_df)

        predict_pipeline = PredictionPipeline()

        result = predict_pipeline.predict(pred_df)
        formatted_result = f"{result[0]:.2f}"  # Format the result to 2 decimal places using an f-string
        print("Prediction: ", formatted_result)
        return render_template('home.html', results=formatted_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug= True, port="8000")