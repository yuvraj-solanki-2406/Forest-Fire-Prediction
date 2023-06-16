from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__, template_folder="templates")

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict_result', methods=['POST', 'GET'])
def predictResult():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]   
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    # output = (int(prediction[0][1]))*100

    if output > str(0.5):
        return render_template('home.html', pred='Your Forest is in Danger.\tProbability of fire occuring is {}'.format(output))
    else:
        return render_template('home.html', pred='Your Forest is safe.\tProbability of fire occuring is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True, port=1212)
