import json
import logging

from flask import Flask, request
from chatbot import ChatBot
from flask_cors import CORS

# TextPreparationVectorizer, NormalizationScaler are needed for pickling of the cuisine_classifier
from ic_components import TextPreparationVectorizer, NormalizationScaler


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.post("/ask")
def ask():

    data = request.json

    try:
        utterance = data["text"]
    except KeyError:
        return json.dumps({"error": "User have not provided any text input."})
    
    answer = bot.ask(utterance)

    try:
        result = json.dumps(answer)
    except TypeError as e:
        return json.dumps({"error": str(e)})
    
    return result


if __name__ == "__main__":

    bot = ChatBot("AlexaBot", path=r"./application", log_level=logging.DEBUG)

    app.run(host="0.0.0.0", port="5000", debug=True)

