from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import pandas as pd

# load dataset
# dataset contains singlish-english conversational pairs
ds = load_dataset("csv", data_files="singlish_to_english_v0.3.csv")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True  # Offload FP32 weights to CPU if necessary
)

# set up QLoRA
lora_config = LoraConfig(
    r=4,  # low-rank factor
    lora_alpha=16,  # scaling parameter
    target_modules=["q_proj", "v_proj"],  # commonly used for attention layers
    lora_dropout=0.1,  # dropout for added regularization
)

# apply QLoRA to the model
model = get_peft_model(model, lora_config)

def preprocess_function(examples):
    inputs = examples['singlish']
    targets = examples['english']
    # format inputs in a conversational format
    model_inputs = tokenizer(inputs, padding="max_length", max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# set pad_token to match eos_token since llama tokenizer does not have a pad_token by default
tokenizer.pad_token = tokenizer.eos_token

# tokenize fine-tune dataset
tokenized_ds = ds.map(preprocess_function, batched=True)

# define training args
training_args = TrainingArguments(
    output_dir="llama-3.2-3B-singlish-finetuned",
    per_device_train_batch_size=2,  # reduce batch size if memory issues occur
    gradient_accumulation_steps=8,  # effective larger batch size
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,  # enable mixed precision for memory efficiency
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100
)

# train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"]
)
trainer.train()

# save the fine-tuned model and tokenizer
trainer.save_model("./llama-3.2-3B-singlish-finetuned")
tokenizer.save_pretrained("./llama-3.2-3B-singlish-finetuned")