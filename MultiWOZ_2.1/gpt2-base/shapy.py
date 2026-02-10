import shap

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

import pandas as pd
from datasets import load_dataset
import torch
import pickle

#===============================================Load Model===============================================================

model_checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained('./gpt2-multiwoz-base')

special_tokens_dict = {'additional_special_tokens': ['[EOC]','[SOC]','[User]','[Bot]', '[SOR]', '[EOR]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(torch.device(device))

#===============================================Load Model===============================================================

df = pd.read_csv('base/train.csv')
context = []
response = []

i=0
train = df['text'].to_list()

for instance in train:
    text = instance.split('[EOC]')[0]+'[EOC]'
    context.append(text)

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    sample_output = model.generate(input_ids, 
                                   do_sample=True, 
                                   max_length=input_ids.shape[1]+50, 
                                   top_k=0, 
                                   pad_token_id=tokenizer.eos_token_id
                                  )

    response.append(tokenizer.decode(sample_output[0][input_ids.shape[1]:], 
                                     skip_special_tokens=False).split('[EOR]')[0])

    i = i+1
    if i==10:
        print(i, "done out of ", len(train))
    if i%100==0:
        print(i, "done out of ", len(train))


df = pd.DataFrame.from_dict({'Context':context, 'Response':response})

teacher_forcing_model = shap.models.TeacherForcing(model, tokenizer)
masker = shap.maskers.Text(tokenizer, mask_token = "...", collapse_mask_token=True)
explainer = shap.Explainer(teacher_forcing_model, masker)

shap_values = explainer(context, response)

with open('shap_values_multiwoz.pk', 'wb') as f:
    pickle.dump(shap_values, f, protocol=pickle.HIGHEST_PROTOCOL)
