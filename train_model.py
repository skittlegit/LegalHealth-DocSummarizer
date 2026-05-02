# Cell 1 — Imports & config
import os
import random
import math
from pprint import pprint
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
import nltk
nltk.download('punkt')
from heapq import nlargest
from collections import defaultdict
import re 
import pandas as pd

# Cell 2 — Settings 
import random
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Model choices
MODEL_NAME = "t5-small"
TOKENIZER_NAME = MODEL_NAME

# Dataset choices
MED_DATASET = "ccdv/pubmed-summarization"
LEGAL_DATASET = "billsum"

# DATASET CONFIGURATION 

TRAIN_SAMPLES_PER_DOMAIN = 500000
VAL_SAMPLES_PER_DOMAIN = 1000
TEST_SAMPLES_PER_DOMAIN = 1000

# Tokenization settings
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

output_dir_path = globals().get("DRIVE_OUTPUT_DIR", "./t5_lora_summarizer_local")

OUTPUT_DIR = output_dir_path
NUM_TRAIN_EPOCHS = 3            
PER_DEVICE_BATCH_SIZE = 2          
GRAD_ACCUM_STEPS = 4               
LEARNING_RATE = 5e-5

print(f"Training configured. \nOutput Directory: {OUTPUT_DIR} \nEpochs: {NUM_TRAIN_EPOCHS}")

# Cell 3 — Load datasets
print("Loading datasets...")

med = load_dataset(MED_DATASET)
legal = load_dataset(LEGAL_DATASET)

# Inspect
print("Medical dataset splits:", med)
print("Legal dataset splits:", legal)

def make_subset(ds, split_name, n):
    if split_name not in ds:
        return None
    size = len(ds[split_name])
    take_n = min(n, size)
    return ds[split_name].select(range(take_n))

# Build combined dataset dict for fine-tuning (concatenate domain datasets)
train_list = []
val_list = []
test_list = []

if "train" in med:
    train_list.append(make_subset(med, "train", TRAIN_SAMPLES_PER_DOMAIN))
if "validation" in med:
    val_list.append(make_subset(med, "validation", VAL_SAMPLES_PER_DOMAIN))
if "test" in med:
    test_list.append(make_subset(med, "test", TEST_SAMPLES_PER_DOMAIN))

if "train" in legal:
    train_list.append(make_subset(legal, "train", TRAIN_SAMPLES_PER_DOMAIN))
if "validation" in legal:
    val_list.append(make_subset(legal, "validation", VAL_SAMPLES_PER_DOMAIN))
if "test" in legal:
    test_list.append(make_subset(legal, "test", TEST_SAMPLES_PER_DOMAIN))

from datasets import concatenate_datasets
train_ds = concatenate_datasets(train_list) if train_list else None
val_ds = concatenate_datasets(val_list) if val_list else None
test_ds = concatenate_datasets(test_list) if test_list else None

print("Train size:", len(train_ds) if train_ds else 0)
print("Val size:", len(val_ds) if val_ds else 0)
print("Test size:", len(test_ds) if test_ds else 0)

# Cell 4 — Inspect sample fields for both datasets and map fields to 'input' and 'target'
print("Example medical sample keys:", med['train'].column_names if 'train' in med else med.column_names)
print("Example legal sample keys:", legal['train'].column_names if 'train' in legal else legal.column_names)

def find_field(ds, candidate_inputs, candidate_targets):
    for c in candidate_inputs:
        if c in ds.column_names:
            for t in candidate_targets:
                if t in ds.column_names:
                    return c, t

    cols = ds.column_names
    return cols[0], cols[1] if len(cols) > 1 else cols[0]

# For med:
if 'train' in med:
    med_in, med_out = find_field(med['train'], ["article","abstract","text"], ["highlights","abstract","summary"])
else:
    med_in, med_out = find_field(med, ["article","abstract","text"], ["highlights","abstract","summary"])

if 'train' in legal:
    leg_in, leg_out = find_field(legal['train'], ["text","article","document"], ["summary","highlights"])
else:
    leg_in, leg_out = find_field(legal, ["text","article","document"], ["summary","highlights"])

print("Medical fields chosen:", med_in, med_out)
print("Legal fields chosen:", leg_in, leg_out)

# For mapping later, record per dataset
MED_INPUT_COL, MED_TARGET_COL = med_in, med_out
LEG_INPUT_COL, LEG_TARGET_COL = leg_in, leg_out

# Cell 5 — Prepare tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# We'll patch the model with LoRA using PEFT
print("Model and tokenizer loaded. Model params:", sum(p.numel() for p in base_model.parameters()))

# Cell 6 — Preprocessing function
def preprocess_function_examples(examples, input_col, target_col):
    inputs = examples[input_col]
    targets = examples[target_col]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply to our combined subsets and produce tokenized datasets
print("Tokenizing datasets...")
tokenized_train = None
try:
    if train_ds is not None:
        tokenized_train = train_ds.map(lambda x: preprocess_function_examples(x, MED_INPUT_COL if MED_INPUT_COL in train_ds.column_names else LEG_INPUT_COL, MED_TARGET_COL if MED_TARGET_COL in train_ds.column_names else LEG_TARGET_COL), batched=True)
except Exception as e:
    print('Simple tokenization mapping failed, will use robust mapping. Error:', e)

def preprocess_mixed(ds, med_cols, leg_cols):
    def fn(batch):
        inputs = []
        targets = []
        for i, _ in enumerate(batch[list(batch.keys())[0]]):
            if med_cols[0] in batch and batch[med_cols[0]][i] is not None:
                in_text = batch[med_cols[0]][i]
                out_text = batch[med_cols[1]][i] if med_cols[1] in batch else ""
            elif leg_cols[0] in batch and batch[leg_cols[0]][i] is not None:
                in_text = batch[leg_cols[0]][i]
                out_text = batch[leg_cols[1]][i] if leg_cols[1] in batch else ""
            else:
                keys = list(batch.keys())
                in_text = batch[keys[0]][i]
                out_text = batch[keys[1]][i] if len(keys)>1 else ""
            inputs.append(in_text)
            targets.append(out_text)
        tokenized = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
        tokenized["labels"] = labels["input_ids"]
        return tokenized
    return fn

if tokenized_train is None and train_ds is not None:
tokenized_train = train_ds.map(preprocess_mixed(train_ds, (MED_INPUT_COL, MED_TARGET_COL), (LEG_INPUT_COL, LEG_TARGET_COL)), batched=True)
tokenized_val = val_ds.map(preprocess_mixed(val_ds, (MED_INPUT_COL, MED_TARGET_COL), (LEG_INPUT_COL, LEG_TARGET_COL)), batched=True) if val_ds else None
tokenized_test = test_ds.map(preprocess_mixed(test_ds, (MED_INPUT_COL, MED_TARGET_COL), (LEG_INPUT_COL, LEG_TARGET_COL)), batched=True) if test_ds else None
print("Tokenization done.")

# Cell 7 — Setup PEFT (LoRA) on model
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]  
)

model = get_peft_model(base_model, peft_config)
print("PEFT/LoRA applied. Trainable params (should be small):", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Cell 8 — Data collator and training arguments
# Create with the most-compatible minimal set of kwargs
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    predict_with_generate=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=True,
    push_to_hub=False,
    report_to="none",
)
# Set these after construction (works even if constructor rejects them)
try:
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "epoch"
except Exception:
    # some builds freeze attributes, but most accept assignment
    pass

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
print("Training args and data collator prepared.")

# Cell 9 — Trainer & train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val if tokenized_val is not None else tokenized_train.select(range(min(200, len(tokenized_train)))),
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training complete — model saved to", OUTPUT_DIR)

# Cell 10: Evaluation
from tqdm import tqdm

# 1. Load Metrics
rouge = evaluate.load("rouge")

def generate_summary(batch_inputs):
    inputs = tokenizer(batch_inputs, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"], 
            max_new_tokens=128, 
            min_length=60,             
            num_beams=4,             
            length_penalty=2.0, 
            early_stopping=True
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\.([a-zA-Z])', r'. \1', text) 
    return text

print("🚀 Starting Smart Evaluation...")
test_subset = test_dataset.select(range(50))  

predicted_summaries = []
reference_summaries = []

for example in tqdm(test_subset):

    input_text = example.get("article") or example.get("text") or ""
    ref_text = example.get("abstract") or example.get("summary") or ""
    
    raw_summary = generate_summary([ "summarize: " + input_text ])[0]
 
    final_summary = clean_text(raw_summary)
    
    predicted_summaries.append(final_summary)
    reference_summaries.append(ref_text)


results = rouge.compute(predictions=predicted_summaries, references=reference_summaries, use_stemmer=True)

final_scores = {k: round(v * 100, 2) for k, v in results.items()}

print("\n✅ SCORES (With Formatting Fixes):")
df = pd.DataFrame([final_scores], index=["Fine-Tuned Model (Smart)"])
print(df)

print("📊 Calculating Baseline Scores...")

def lead3(text):
    return " ".join(re.split(r'(?<=[.?!])\s+', str(text))[:3])
    
baseline_summaries = [lead3(text) for text in test_subset["article"]] 

base_results = rouge.compute(predictions=baseline_summaries, references=reference_summaries, use_stemmer=True)
print("\n🏆 BASELINE (Lead-3) SCORES:")
print({k: round(v * 100, 2) for k, v in base_results.items()})

print("\n------------------------------------------------")

def simple_textrank(text, num_sentences=3):
    sentences = re.split(r'(?<=[.?!])\s+', str(text))
    if len(sentences) <= num_sentences:
        return str(text)
 
    scores = [0] * len(sentences)
    for i, s1 in enumerate(sentences):
        words1 = set(s1.lower().split())
        if not words1: continue
        for j, s2 in enumerate(sentences):
            if i == j: continue
            words2 = set(s2.lower().split())
            if not words2: continue
 
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            scores[i] += intersection / union if union > 0 else 0

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    top_indices.sort()
    
    return " ".join([sentences[i] for i in top_indices])

# 2. Run Evaluation
print("📊 Calculating TextRank Scores...")
rouge = evaluate.load("rouge")
textrank_preds = []
for item in tqdm(test_subset):
    input_text = item.get("article") or item.get("text") or ""
    summary = simple_textrank(input_text)
    textrank_preds.append(summary)

# 3. Compute Scores
tr_results = rouge.compute(predictions=textrank_preds, references=reference_summaries, use_stemmer=True)

print("\n🏆 TextRank SCORES:")
print({k: round(v * 100, 2) for k, v in tr_results.items()})
