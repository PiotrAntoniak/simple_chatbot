import pkg_resources
from symspellpy import SymSpell, Verbosity
from sentence_transformers import SentenceTransformer, util
import numpy as np 
import flask
from flask import Flask
from flask import request
app = Flask(__name__)

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.cpu()

inp_emb_dicto = np.load('inp_emb_dicto.npy',allow_pickle='TRUE').item()
inp_res_dicto = np.load('inp_res_dicto.npy',allow_pickle='TRUE').item()

chat = ""


with open("DO_NOT_CHECK.txt","r") as f:
    DO_NOT_CHECK = f.read()
    DO_NOT_CHECK.split(" ")
     
@app.route('/')
def home():
    return flask.render_template('index.html')

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

@app.route('/match_response',methods=['POST'])
def match_response():
    global chat
    
    sentence =  [x for x in request.form.values()]
    sentence = sentence[0]
    embeddings1 = model.encode(" ".join(fix_sentence(sentence)))
    best_score = 0
    best_key = None
    
    for key in inp_emb_dicto.keys():
        temp_score = util.pytorch_cos_sim(embeddings1, inp_emb_dicto[key])
        if temp_score > best_score:
            best_key = key
            best_score = temp_score
    print(best_score)
    print(best_key)
    if best_score > 0.5:
        answer = inp_res_dicto[best_key]
    else:
        answer = "I'm sorry, I did't quite get what you want. I'm connecting you to one of our consultants."
        
    chat = chat + "Q: " + sentence + "<br>"
    chat = chat + "A: "+ str(answer) + "<br>" + "<br>"
    return flask.render_template('index.html',chat = chat)

if __name__ == '__main__':
    app.run()
   