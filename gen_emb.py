# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 21:52:16 2021

@author: pix
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('config')
loaded_quantized_model =  torch.jit.load( "quant/pytorch_model.pth").cpu()
loaded_quantized_model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['token_embeddings'] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def calculate_emb(sentence, model = loaded_quantized_model, tokenizer = tokenizer):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    features = {'input_ids':encoded_input['input_ids'], 'attention_mask':encoded_input['attention_mask']}
    with torch.no_grad():
        output = model(features)

    sentence_embeddings = mean_pooling(output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def main():
    data = pd.read_csv("input and responses.csv", low_memory=False)

    inp_res_dicto = {}
    for inp,res in zip(data.INPUT_TEXT, data.RESPONSE):
        inp_res_dicto.update({inp:res})
        np.save('inp_res_dicto.npy', inp_res_dicto) 

    inp_emb_dicto = {}
    for i,inp in enumerate(data.INPUT_TEXT.tolist()):
        inp_emb_dicto.update({inp:calculate_emb(inp)})
        np.save('inp_emb_dicto.npy', inp_emb_dicto) 

if __name__ == '__main__':
    main()