import subprocess
import sys
import os
import torch
from gtts import gTTS
from transformers import DistilBertTokenizer, DistilBertModel, GPT2LMHeadModel, GPT2Tokenizer

# Print a welcome message and ASCII art
print("Bob is here. Type 'quit' to exit.\n")
print("  BBBBBBBB    OOOOOOOOOO    BBBBBBBB      GGGGGG      PPPPPPPP  TTTTTTTTTT")
print("  BB    BB  OOOO      OOOO  BB    BB    GG      GG    PP      PP    TT    ")
print("  BB    BB  OO          OO  BB    BB  GG              PP      PP    TT    ")
print("  BBBBBBBB  OO          OO  BBBBBBBB  GG    GGGGGG    PP      PP    TT    ")
print("  BB    BB  OO          OO  BB    BB  GG        GG    PPPPPPPP      TT    ")
print("  BB    BB  OOOO      OOOO  BB    BB  GGGG      GG    PP            TT    ")
print("  BBBBBB      OOOOOOOOOO    BBBBBB        GGGGGGGG    PP            TT    ")

# Auto-install function
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install gtts if needed
def ensure_gtts_installed():
    try:
        import gtts
    except ImportError:
        print("gTTS not found. Installing...")
        install("gtts")
        import gtts

# Ensure necessary libraries are installed
def try_import(module, package_name):
    try:
        __import__(module)
    except ModuleNotFoundError:
        print(f"Required module '{package_name}' not found. Installing '{package_name}'...")
        install(package_name)

# Ensure libraries are installed
try_import('torch', 'torch')
try_import('transformers', 'transformers')
ensure_gtts_installed()

# Text Generation with DistilBERT as Encoder and GPT-2 as Decoder
def generate_text(prompt):
    # Initialize tokenizer and models
    bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Tokenize the prompt with DistilBERT
    inputs = bert_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    # Get the hidden states from DistilBERT
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Use the last hidden state as input to GPT-2
    gpt2_inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    gpt2_outputs = gpt2_model.generate(input_ids=gpt2_inputs, max_length=1024, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    # Decode the output from GPT-2
    generated_text = gpt2_tokenizer.decode(gpt2_outputs[0], skip_special_tokens=True)
    
    return generated_text

# Function to speak text using gTTS
def speak_gtts(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    
    # Use subprocess to play the sound, works more reliably
    if sys.platform == "win32":
        subprocess.Popen(["start", "response.mp3"], shell=True)  # Windows
    elif sys.platform == "darwin":
        subprocess.Popen(["afplay", "response.mp3"])  # macOS
    else:
        subprocess.Popen(["mpg321", "response.mp3"])  # Linux (ensure mpg321 is installed)

# Loop to continue conversation with Bob
while True:
    prompt = input("Ask Bob: ")
    if prompt.lower() == "quit":
        print("Ending conversation with Bob.")
        break

    generated_text = generate_text(prompt)
    
    # Print and read aloud Bob's response
    print("Bob says:", generated_text)
    speak_gtts(generated_text)  # Use gTTS to read the text out loud
