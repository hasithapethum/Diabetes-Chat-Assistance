import json
import random
import datetime
import torch
import os
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import torch

# Load data from the data.pth file
data = torch.load("data.pth")

# Extract input_size, hidden_size, and output_size from the loaded data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Now you can proceed to instantiate the model using these values
model = NeuralNet(input_size, hidden_size, output_size).to(device)

csv_folder = "csv_data"  # Specify the folder containing CSV files

# Function to load CSV files from the folder
def load_csv_files():
    csv_data = {}
    for file_name in os.listdir(csv_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(csv_folder, file_name)
            df = pd.read_csv(file_path, encoding='latin1')  # Specify encoding as latin1
            csv_data[file_name.split('.')[0]] = df  # Store dataframes with file names as keys
    return csv_data


# Load CSV files
csv_data = load_csv_files()

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Diabetes Assistance"

# Define the path to the log file
log_file = "unanswered_questions.log"

# Cache to store answers
answer_cache = {}

# Function to split response into text and link parts
def split_response(response):
    if "https://" in response:
        parts = response.split("https://")
        if len(parts) > 1:
            text_part = parts[0]
            link_part = "https://" + parts[1]
        else:
            text_part = response
            link_part = None
    else:
        text_part = response
        link_part = None
    return text_part, link_part

# Function to log unanswered questions
def log_unanswered_question(question):
    with open(log_file, "a") as log:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{timestamp} - Unanswered Question: {question}\n")

# Function to find the answer for a given question
def find_answer(user_question):
    user_question = user_question.lower()
    if user_question in answer_cache:
        return answer_cache[user_question]
    return None

# Function to get a response based on user input
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                text_part, link_part = split_response(response)
                if link_part:
                    response = f'<a href="{link_part}">{text_part}</a>.'
                return response

    answer = find_answer(msg)
    if answer:
        text_part, link_part = split_response(answer)
        if link_part:
            answer = f'<a href="{link_part}">{text_part}</a>.'
        return answer

    # If no response was found in intents or cache, log the unanswered question
    log_unanswered_question(msg)

    return "I couldn't find an answer to your question. Please ask questions related to Diabetes."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        question = input("User : ")
        if question.lower() == "quit":
            break
        response = get_response(question)
        print("Bot : ", response)
