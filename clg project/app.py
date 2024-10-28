import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import inputScript

# Load models and vectorizer
model1 = pickle.load(open('phishing.pkl', 'rb'))  # Phishing detection model
model2 = pickle.load(open('mail.pkl', 'rb'))       # Email spam detection model

# Load and preprocess email data for feature extraction
raw_mail = pd.read_csv("fraud_email_.csv")
mail_data = raw_mail.where((pd.notnull(raw_mail)), '')
mail_data.loc[mail_data['Class'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Class'] == 'ham', 'Category'] = 1

X = mail_data['Text']
Y = mail_data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
feature_extraction.fit(X_train)

app = Flask(__name__, template_folder='templates', static_folder='staticFiles')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_phishing():
    if request.method == 'POST':
        url = request.form.get('URL')
        if not url:
            return render_template('predict.html', prediction="Error: URL is required")

        try:
            # Assuming inputScript.Phishing_Website_Detection is defined somewhere
            checkprediction = inputScript.Phishing_Website_Detection(url)
            checkprediction = np.array(checkprediction).reshape(1, -1)
            prediction = model1.predict(checkprediction)
            output = prediction[0]
            if output == 1:
                pred = "Safe! This is a Legitimate Website."
            else:
                pred = "Suspicious. Be cautious!"
            return render_template('predict.html', prediction=pred, url="The URL is: " + url)
        except Exception as e:
            return render_template('predict.html', prediction='An error occurred: {}'.format(e), url="The URL is: " + url)
    else:
        return render_template('predict.html', prediction='Error: Invalid request method')

@app.route('/sendd', methods=['POST'])
def predict_email():
    a = request.form.get('mail_check')
    c = request.form.get('out_type')

    if not a:
        return "Enter some value of mail"
    if c is None:
        return "Select the output type"

    input_mail = [a]
    input_mail = feature_extraction.transform(input_mail)
    result = model2.predict(input_mail)

    if c == "Json_format":
        status = "Warning! It's a spam message." if result[0] == 0 else "It's a safe message."
        return jsonify({"Message": a, "Ans_format": c, "Status": status})

    return render_template('predict1.html', label=result[0], message=a)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        if not data:
            raise ValueError("Invalid JSON data")
        input_array = np.array(list(data.values()))
        input_array = input_array.reshape(1, -1)
        prediction = model1.predict(input_array)
        output = prediction[0]
        return jsonify({'output': output})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
