import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import warnings

# Ignore specific warnings by filtering them
warnings.filterwarnings("ignore")




class BertSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertSentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout1(pooled_output)
        output = self.fc1(pooled_output)
        output = self.dropout2(output)
        logits = self.fc2(output)
        
        return logits



# # create an instance of the BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

# bert_model_name = "bert-base-uncased"
# num_classes = 6
# model_predict = BertSentimentClassifier(bert_model_name, num_classes) #("bert-base-uncased", 6)
# model_predict.load_state_dict(torch.load("weights_epoch10.pt", map_location=torch.device('cpu'))) 


# input_text = "Ben love My"
# batch_size = 32
# max_len = 150
# device = "cpu"
# labels = ["Anger","Fear","Joy","Love","Sadness","Surprise"]
# le = LabelEncoder()
st.set_page_config(page_title="Emotion Classification", page_icon="logo_csie2.png")
st.image("title.png")
st.sidebar.image("logo_NCKU.jpeg", use_column_width=True)
# # Tạo chân trang bằng cách sử dụng HTML
# footer_html = """
# <div style="background-color: #f0f0f0; padding: 10px; text-align: center;">
#     <p>&copy; 2023 Ben Phan - CSIE - NCKU </p>
# </div>
# """

footer_html = """
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f0f0f0; padding: 10px; text-align: center;">
    <p>&copy; 2023 2023 Ben Phan - CSIE - NCKU</p>
</div>
"""




# Create a function to load the BERT model
@st.cache(allow_output_mutation=True)
def load_bert_model():
    bert_model_name = "bert-base-uncased"
    num_classes = 6
    model_predict = BertSentimentClassifier(bert_model_name, num_classes)
    model_predict.load_state_dict(torch.load("weights_epoch10.pt", map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    return model_predict, tokenizer

# Load the model only once
model_predict, tokenizer = load_bert_model()

# The rest of your Streamlit app code
input_text = "Ben love My"
batch_size = 32
max_len = 150
device = "cpu"
labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] 

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    if prompt == "Hello" or prompt == "hello" or prompt == "Hi" or prompt == "hi":
        response = f"IIR Bot: Hello 👋, I am a language model, I can help you analyze the emotional state of a sentence." 
    elif prompt == "good bye" or prompt == "Good bye" or prompt == "Goodbye" or prompt == "goodbye":
        response = f"IIR Bot: Good bye 👋, see you next time." 
    elif prompt == "bye" or prompt == "Bye":
        response = f"IIR Bot: Good bye 👋, see you next time."  
    else:    
        encoded_input = tokenizer.encode_plus(
            prompt, 
            add_special_tokens=True,
            truncation=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)
        emotion_id = torch.argmax(model_predict(**encoded_input), dim=1).item()

        if labels[emotion_id] == "anger": 
            response = f"IIR Bot: The above sentence seems a bit anger 😠 ."
        elif labels[emotion_id] == "fear":
            response = f"IIR Bot: The above sentence conveys a sense of fear 😨 . "
        elif labels[emotion_id] == "joy":
            response = f"IIR Bot: The above sentence exudes a feeling of joy 😄 . "
        elif labels[emotion_id] == "love":
            response = f"IIR Bot: The above sentence conveys a deep sense of love 😍 . "
        elif labels[emotion_id] == "sadness":
            response = f"IIR Bot: The above sentence expresses profound sadness 😢 . "
        elif labels[emotion_id] == "surprise":
            response = f"IIR Bot: The above sentence evokes a sense of surprise 😲 . "

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
