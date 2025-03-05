**Diamond Price Prediction**

GOAL:
    The goal is to predict the PRICE of a diamond.
ABOUT DATASET:
      This classic dataset contains the prices and other attributes of almost 54,000 diamonds.
  The independent variables:
      •	carat : the unique unit of weight measurement used exclusively to weigh the diamonds(0.2-5.01)
      •	cut : quality of Diamond cut(Fair, Good, Very Good, Premium, Ideal)
      •	color : diamond color, from J(worst) to D(best)
      •	clarity : a measurement of how clear the diamond is (I1(worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF(best))
      •	depth : The depth of diamond is its height.
      total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43-79)
      •	table : width of top of diamond relative to widest point(43-95)
      •	x : X dimension
      •	y : Y dimension
      •	z : Z dimension
  Target variable:
      •	price: price in US dollars (\$326--\$18,823)
  •	Dataset Source Link  
    https://www.kaggle.com/datasets/lovishbansal123/diamond-dataset
AWS Deployment Link :

AWS Elastic Beanstalk link :  

Approach for the project
    1.	Data Ingestion :
        o	In Data Ingestion phase the data is first read as csv.
        o	Then the data is split into training and testing and saved as csv file.
    2.	Data Transformation :
        o	In this phase a ColumnTransformer Pipeline is created.
        o	for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
        o	for Categorical Variables SimpleImputer is applied with most frequent strategy, then performed ordinal encoding, after this data is scaled with Standard Scaler.
        o	This preprocessor is saved as pickle file.
    3.	Model Training :
        o	In this, base model is tested . The best model found was catboost, however picked the xgboost model with similar output and very slight difference.
        o	After this hyperparameter tuning is performed on xgboost.
        o	This model is saved as pickle file.
    4.	Predict Pipeline :
        o	This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.
    5.	Flask App :
        o	Flask app is created with User Interface to predict the Diamond prices inside a Web Application.
Exploratory Data Analysis Notebook
Link : 

