from flask import Flask, render_template, request, redirect, url_for
import json
import subprocess
from flask import render_template
import os

# Get the path to the folder containing app.py (admin folder)
current_folder = os.path.dirname(__file__)

# Construct the path to abc.json using the parent directory
json_file_path = os.path.join(current_folder, "..", "intents.json")

train_pth= os.path.join(current_folder, "..", "train.py")

app = Flask(__name__)

# Load the initial dataset
with open(json_file_path, 'r') as file:
    dataset = json.load(file)

@app.route('/')
def index():
    return render_template('index.html', intents=dataset['intents'])

@app.route('/update_intent/<tag>', methods=['GET', 'POST'])
def update_intent(tag):
    if request.method == 'POST':
        new_response = request.form['new_response']
        for intent in dataset['intents']:
            if intent['tag'] == tag:
                intent['responses'].append(new_response)
        with open('dataset.json', 'w') as file:
            json.dump(dataset, file, indent=2)
        return redirect(url_for('index'))

    return render_template('update_intent.html', tag=tag)

@app.route('/delete_intent/<tag>', methods=['GET', 'POST'])
def delete_intent(tag):
    if request.method == 'POST':
        dataset['intents'] = [intent for intent in dataset['intents'] if intent['tag'] != tag]
        with open('dataset.json', 'w') as file:
            json.dump(dataset, file, indent=2)
        return redirect(url_for('index'))

@app.route('/create_intent', methods=['GET', 'POST'])
def create_intent():
    if request.method == 'POST':
        tag = request.form['tag']
        patterns = [pattern.strip() for pattern in request.form['patterns'].split(',')]
        responses = [response.strip() for response in request.form['responses'].split(',')]
        new_intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }
        dataset['intents'].append(new_intent)
        with open('dataset.json', 'w') as file:
            json.dump(dataset, file, indent=2)
        return redirect(url_for('index'))

    return render_template('create_intent.html')

@app.route('/search', methods=['POST'])
def search():
    search_term = request.form['search_term'].lower()
    search_results = [intent for intent in dataset['intents'] if search_term in intent['tag'].lower()]
    return render_template('index.html', intents=search_results)

@app.route('/run_training', methods=['GET', 'POST'])
def run_training():
    if request.method == 'POST':
        # Run the "train.py" script and capture its output
        result = subprocess.run(['python', train_pth], capture_output=True, text=True)
        # Pass the result to the template
        return render_template('training_results.html', result=result.stdout)

    return render_template('run_training.html')


if __name__ == '__main__':
    app.run(debug=True,port=5010)
