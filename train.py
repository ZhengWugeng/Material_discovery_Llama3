# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
import argparse

# %%
# !huggingface-cli login --token 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

# %%
model_name = 'meta-llama/Meta-Llama-3-8B'
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             #load_in_8bit=True,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto"
                                            )
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
data = load_dataset("ZWG817/Full_Content_Train")
data_train = data["train"]
print(data_train)

#custom_data = load_dataset('json', data_files='data_eval.json')
#data_val = custom_data['train']

with open('materials.txt', 'r') as file:
    word_list = file.read().splitlines()

# %%
def generate_prompt(dialogue, summary=None, eos_token="</s>"):
  instruction = "In your role as a helpful scientific assistant, Read the following text and summarize the following literature. At the same time, please make sure molecular formulas are involved:\n "
  input = f"{dialogue}\n "
  summary = f"Summary of literature:\n {summary + ' ' + eos_token if summary else ''} "
  prompt = (" ").join([instruction, input, summary])
  return prompt

print(generate_prompt(data_train[4]['abstract']))

# %%
input_prompt = generate_prompt(data_train[6]['content'][:20000])
input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
with torch.cuda.amp.autocast():
  generation_output = model.generate(
      input_ids=input_tokens,
      max_new_tokens=1000,
      do_sample=True,
      top_k=10,
      top_p=0.9,
      temperature=0.3,
      repetition_penalty=1.15,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.eos_token_id,
    )
# op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
# print(op)

# %%
lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

# %%
print(model)

# %%
# this should be set for finutning and batched inference
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

# %%
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# %%
output_dir = "cp"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
per_device_eval_batch_size = 1
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 200
logging_steps = 10
learning_rate = 5e-4
max_grad_norm = 0.3
max_steps = 200
warmup_ratio = 0.03
evaluation_strategy="steps"
lr_scheduler_type = "constant"

training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            evaluation_strategy=evaluation_strategy,
            save_steps=save_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            max_grad_norm=max_grad_norm,
            # max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            ddp_find_unused_parameters=False,
            eval_accumulation_steps=eval_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
        )

# %%
data_val = data_train.select(range(200))

# %%
def formatting_func(prompt):
  output = []

  for d, s in zip(prompt["content"][:16384], prompt["abstract"]):
    op = generate_prompt(d, s)
    output.append(op)

  return output

trainer = SFTTrainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_val,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1800,
    tokenizer=tokenizer,
    args=training_args
)

# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model(f"{output_dir}/final")

# %%
model.save_pretrained('result', save_embedding_layers=True)

# %%
model.push_to_hub("ZWG817/Llama3_Chat_Materials")
tokenizer.push_to_hub("ZWG817/Llama3_Chat_Materials")

# %%



