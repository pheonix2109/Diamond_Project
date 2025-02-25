from flask import Flask, request, render_template,jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
# @cross_origin()
def home_page():
    return render_template('home.html')

@app.route('/home',methods=['GET','POST'])
# @cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        pred_df = data.get_data_as_dataframe()
        
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug= True)