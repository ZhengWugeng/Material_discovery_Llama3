# %%
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import pipeline
import numpy as np
import pandas as pd
import torch

# %%
model = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


# %%
# model_name = "meta-llama/Meta-Llama-3-8B"
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              torch_dtype=torch.bfloat16,
#                                              device_map="auto"
#                                             )
# tokenizer = AutoTokenizer.from_pretrained('ZWG817/Llama3_Chat_Materials')
# model.resize_token_embeddings(len(tokenizer))

# model.load_adapter('ZWG817/Llama3_Chat_Materials')

# %%
data = load_dataset("ZWG817/Materials_Gilbert_Damping")
data_train = data["train"]

# %%
system_prompt = "You are a helpful assistant. Read the following text and determine if it mentions the Gilbert damping constant of any material. If it does, list each material's molecular formula and its corresponding Gilbert damping constant. Please format your answer as follows:\nChemical Formula: [Formula]\nGilbert Damping Constant: [Value]\nOriginal Sentences: [Sentences]\nIf the text does not mention the Gilbert damping constant, please respond with:\nNo Mention"

# %%
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": data_train[4]['content'][:20000]},
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
assistant_response = outputs[-1]["generated_text"][-1]["content"]
print(assistant_response)

# %%
df = pd.DataFrame()
df_tmp = pd.DataFrame()
df['Damping Constant'] = []
results = []

for i,j in enumerate(data_train):
    print(i)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data_train[i]['content'][:20000]},
    ]
    
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        assistant_response = outputs[-1]["generated_text"][-1]["content"]
        print(assistant_response)
        results.append(assistant_response)
        
    except:
        pass

    if len(results) % 500 == 0:
        df_tmp['Damping Constant'] = pd. Series(results)
        df = pd.concat([df,df_tmp])
        results = []
        df_tmp = pd.DataFrame()
        df = df.reset_index()[['Damping Constant']]
        df.to_json('Gilbert Damping Constant'+str(i)+'.json')
        # df.to_csv('Gilbert Damping Constant.csv',index = False)
    
    # if i >= 20:
    #     break
    
df_tmp['Damping Constant'] = results
df = pd.concat([df,df_tmp])
df = df.reset_index()[['Damping Constant']]
df.to_json('Gilbert Damping Constant.json')
# df.to_csv('Gilbert Damping Constant.csv',index = False)
    

    
    

# %%
df

# %%
model.save_pretrained('result', save_embedding_layers=True)


