from flask import Flask, render_template, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the LLaMA 3 model and tokenizer
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
hf_token = os.getenv('HUGGINGFACE_TOKEN')

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

@app.route("/", methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        inputs = tokenizer.encode(user_input, return_tensors='pt')
        outputs = model.generate(inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return render_template('chat.html', user_input=user_input, response=response)
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
