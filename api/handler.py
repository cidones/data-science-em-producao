import pickle
import pandas as pd 

from flask    import Flask, request, Response

from rossmann.Rossmann import Rossmann

#loading model
model = pickle.load(open('/Users/cidones/data-science-em-producao/model/model_rossmann.pkl', 'rb'))

app = Flask( __name__ )

@app.route('/' )
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # has data
        if isinstance( test_json, dict ): #Example
             test_raw = pd.DataFrame( test_json , index=[0] )      
        else: # multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys() )    
            
        # instantiate rossmann class
        pipeline = Rossmann()
        
        #data cleaning
        df1 = pipeline.data_cleaning ( test_raw )
        
        #feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation ( df3 )
        
        #prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')