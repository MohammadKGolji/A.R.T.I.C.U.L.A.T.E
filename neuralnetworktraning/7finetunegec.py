import os
os.environ["HF_HOME"] = r"C:\Users\mysti\OneDrive\Desktop\angfnweigow\hf_cache"#pls use your own path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback)
import evaluate

MODEL_NAME     = "google/flan-t5-base"
CSV_PATH       = "gec_pairs.csv"
OUTPUT_DIR     = "./gec-model"
TRAIN_SPLIT    = 0.90
SEED           = 42

BATCH_SIZE          = 8
GRAD_ACCUM          = 4
LEARNING_RATE       = 3e-4
WARMUP_STEPS        = 300
MAX_TRAIN_STEPS     = 3000
EVAL_STEPS          = 200
MAX_SOURCE_LENGTH   = 128
MAX_TARGET_LENGTH   = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['source', 'target'])
df = df[df['source'].str.strip() != '']
df = df[df['target'].str.strip() != '']
df = df[df['source'].str.lower().str.strip() != df['target'].str.lower().str.strip()]

print(f"Total pairs loaded: {len(df)}")
print(df['pair_type'].value_counts().to_string())

echo_df  = df[df['pair_type'] == 'echo_recast']
synth_df = df[df['pair_type'] != 'echo_recast']
# Upsample for the echo/recast examples.
df_weighted = pd.concat([echo_df, echo_df, echo_df, synth_df], ignore_index=True)
df_weighted = df_weighted.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"After upsampling echo pairs (3x): {len(df_weighted)} total rows")

def load_jfleg():
    try:
        print("  Downloading JFLEG dataset from HuggingFace...")
        jfleg = load_dataset("jhu-clsp/jfleg", trust_remote_code=True)
        rows = []
        for split in ['validation', 'test']:
            if split not in jfleg:
                continue
            for item in jfleg[split]:
                src = item['sentence'].strip()
                for correction in item['corrections']:
                    tgt = correction.strip()
                    if tgt and tgt.lower() != src.lower():
                        rows.append({'source': src, 'target': tgt,
                                    'pair_type': 'jfleg', 'corpus': 'JFLEG'})
        jfleg_df = pd.DataFrame(rows).drop_duplicates(subset=['source', 'target'])
        print(f"  JFLEG pairs loaded: {len(jfleg_df)}")
        return jfleg_df
    except Exception as e:
        print(f"  Could not load JFLEG ({e}). Continuing with TalkBank data only.")
        return pd.DataFrame()

jfleg_df = load_jfleg()
if not jfleg_df.empty:
    # Mix in external GEC data just to gover more grammar.
    df_weighted = pd.concat([df_weighted, jfleg_df], ignore_index=True)
    df_weighted = df_weighted.sample(frac=1, random_state=SEED).reset_index(drop=True)

n          = len(df_weighted)
split_idx  = int(n * TRAIN_SPLIT)
train_df   = df_weighted.iloc[:split_idx].reset_index(drop=True)
eval_df    = df_weighted.iloc[split_idx:].reset_index(drop=True)

# Keep a small echo-heavy slice in eval to track that specific behavior.
echo_eval = df_weighted[df_weighted['pair_type'] == 'echo_recast'].tail(min(100, len(echo_df) // 5))
eval_df   = pd.concat([eval_df, echo_eval]).drop_duplicates().reset_index(drop=True)

def preprocess(examples):
    inputs = ["Fix grammar: " + s for s in examples['source']]
    targets = examples['target']

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = Dataset.from_pandas(train_df[['source', 'target']]).map(
    preprocess, batched=True, remove_columns=['source', 'target']
)
eval_dataset = Dataset.from_pandas(eval_df[['source', 'target']]).map(
    preprocess, batched=True, remove_columns=['source', 'target']
)
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [p.replace("Fix grammar: ", '').strip() for p in pred_str]

    bleu = bleu_metric.compute(
        predictions=pred_str,
        references=[[l] for l in label_str]
    )

    exact = sum(p.lower().strip() == l.lower().strip()
                for p, l in zip(pred_str, label_str)) / max(len(pred_str), 1)

    # Print a few decoded outputs for quick sanity checks during eval.
    for i in range(min(3, len(pred_str))):
        print(f"  SRC: (see training data) | TGT: {label_str[i]} | PRED: {pred_str[i]}")

    return {
        "bleu":        round(bleu['score'], 2),
        "exact_match": round(exact, 4),
    }


version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
# Use the right eval arg name for older vs newer Transformers versions.
eval_strategy_key = "eval_strategy" if version >= (4, 41) else "evaluation_strategy"

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    fp16=False,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_TRAIN_STEPS,
    **{eval_strategy_key: "steps"},
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    push_to_hub=False,
    dataloader_num_workers=0,
)

try:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
except TypeError:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

print(f"  Max steps:   {MAX_TRAIN_STEPS}")
device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
print(f"  Device:      {device_str}")

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"GEC model saved to {OUTPUT_DIR}/")
