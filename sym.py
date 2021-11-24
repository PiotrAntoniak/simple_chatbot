import pkg_resources
from symspellpy import SymSpell, Verbosity
import numpy as np 
import flask
from flask import Flask
from flask import request
import time
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
app = Flask(__name__)

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

model =  torch.jit.load( "quant/pytorch_model.pth").cpu()
model.eval()

inp_emb_dicto = np.load('inp_emb_dicto.npy',allow_pickle='TRUE').item()
inp_res_dicto = np.load('inp_res_dicto.npy',allow_pickle='TRUE').item()

chat = ""

with open("DO_NOT_CHECK.txt","r") as f:
    DO_NOT_CHECK = f.read()
    DO_NOT_CHECK.split(" ")
     
def fix_sentence(sentence,DO_NOT_CHECK=DO_NOT_CHECK):
    fixed = []
    for word in sentence.split(" "):
        word = word.lower()
        if word in DO_NOT_CHECK:
            fixed.append(word)
        else:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3, include_unknown=True)
            fixed.append(suggestions[0].__str__().split(",")[0])
    return fixed

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
def cos_sim(emb1,emb2, cos = cos):
    return cos(emb1,emb2)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['token_embeddings'] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_emb(sentence, model = model, tokenizer = tokenizer):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    features = {'input_ids':encoded_input['input_ids'], 'attention_mask':encoded_input['attention_mask']}
    with torch.no_grad():
        output = model(features)

    sentence_embeddings = mean_pooling(output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/match_response',methods=['POST'])
def match_response():
    global chat

    sentence =  [x for x in request.form.values()]
    start = time.time()
    sentence = sentence[0]
    fixed = " ".join(fix_sentence(sentence))
    print(fixed)
    embeddings1 = calculate_emb(sentence)
    
    responses = []
    for key in inp_emb_dicto.keys():
        temp_emb = inp_emb_dicto[key]
        temp_score = cos_sim(embeddings1, temp_emb)
        responses.append([inp_res_dicto[key], temp_score.item()])
    ordered_responses = sorted(responses, key=lambda x:x[1])
    
    best_response_score = ordered_responses[-1][1]
    best_response = ordered_responses[-1][0]
    
    #ADD SLEEP TO MIMIC HUMAN
    
    if best_response_score > 0.8:
        answer = best_response
        print(best_response_score, best_response)
    elif best_response_score > 0.6:
        answer = best_response + "<br>" + ordered_responses[-2][0]
        print(best_response_score, best_response)
        print(ordered_responses[-2][1], ordered_responses[-2][0])
    else:
        answer = "I'm sorry, I did't quite get what you want.<br> Please reformulate your question or ask for human help."
        print(best_response_score,"no match found")
        
    chat = chat + "Q: " + sentence + "<br>"
    chat = chat + "A: "+ str(answer) + "<br>" + "<br>"
    return flask.render_template('index.html',chat = chat + "<br>" + str(time.time() - start))

if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=8080)
    app.run()