from flask import Flask, render_template, request, session, jsonify
import openai
from dotenv import load_dotenv
import os
import logging

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

logging.basicConfig(level=logging.DEBUG)


openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route("/", methods=['GET', 'POST'])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('chat.html', chat_history=session['chat_history'])

@app.route("/chat", methods=['POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
        
    user_input = request.form['user_input']
    try:
        session['chat_history'].append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=session['chat_history']
        )
        response_text = response.choices[0].message['content']
        
        session['chat_history'].append({"role": "assistant", "content": response_text})
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        response_text = f"An error occurred: {e}"
        session['chat_history'].append({"role": "assistant", "content": response_text})
    
    return jsonify(session['chat_history'])

if __name__ == '__main__':
    app.run(debug=True)
