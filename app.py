from flask import Flask, request, jsonify
from main.main import predictDisease
import re
import warnings
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


def separate_specialization(sentence):

    pattern = r'(.+?)\((.*?)\)'
    
    match = re.match(pattern, sentence)
    
    if match:
        # Extract the parts of the sentence
        before_bracket = match.group(1).strip()
        inside_bracket = match.group(2).strip()
        
        result = {
            "prediction": before_bracket,
            "specialization": inside_bracket
        }
        
        return result
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    
    # Check if the data is in the expected format
    if 'symptoms' not in data or not isinstance(data['symptoms'], list):
        return jsonify({'error': 'Invalid input format'}), 400

    warnings.filterwarnings("ignore", category=UserWarning)
    
    list_of_symptoms = data['symptoms']
    test_symptoms = ",".join(list_of_symptoms)
    test_predictions = predictDisease(test_symptoms)
    result = separate_specialization(test_predictions["final_prediction"])
    
    return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)