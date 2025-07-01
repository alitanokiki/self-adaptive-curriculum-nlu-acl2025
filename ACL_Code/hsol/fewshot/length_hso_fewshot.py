import math
from sklearn.metrics import precision_recall_fscore_support
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    BertModel, BertTokenizer, BertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM,
    DistilBertTokenizer, DistilBertForMaskedLM,
    AdamW, get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
    set_seed as hf_set_seed
)

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'roberta', 'distilroberta'], help='Choose model: bert, roberta, or distilroberta')
parser.add_argument('--seed', type=int, default=66, help='Random seed')
args = parser.parse_args()
set_seed(args.seed)
seed = args.seed
learning_rate = 1e-5
num_epochs = 5
batch_size = 16
data = 'hso'
strategie = 'length_cut'
method = f'1e5_5_16_{strategie}_{args.model_type}_seed{args.seed}'

if args.model_type == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
elif args.model_type == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
elif args.model_type == 'distilroberta':
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base') 
    model = RobertaForMaskedLM.from_pretrained('distilroberta-base')
else:
    raise ValueError(f"Unsupported model_type: {args.model_type}")

label_words = {
    0: ['hateful'],
    1: ["offensive"],
    2: ["neutral"],
}

dataset = load_dataset('tdavidson/hate_speech_offensive')


def tokenize_and_add_prompt(examples):
    mask_token = tokenizer.mask_token
    prompt_sentences = [sentence + f", this was {mask_token}." for sentence in examples['tweet']]
    return tokenizer(prompt_sentences, padding=False, truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_and_add_prompt, batched=True)
train_dataset = tokenized_dataset['train']
def filter_mask_samples(dataset, tokenizer):
    """
    Filter samples containing the MASK token.
    """
    mask_token_id = tokenizer.mask_token_id
    mask_samples = []
    for sample in tqdm(dataset, desc="Filtering samples"):
        if mask_token_id in sample['input_ids']:
            mask_samples.append(sample)
    
    # Create a new dataset from the filtered samples
    filtered_dataset = {key: [sample[key] for sample in mask_samples] for key in mask_samples[0].keys()}
    return Dataset.from_dict(filtered_dataset)

train_dataset = filter_mask_samples(train_dataset, tokenizer)

verbalizer_ids = {}
for label, words in label_words.items():
    ids = []
    for word in words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        ids.extend(word_ids)
    verbalizer_ids[label] = ids

labels = train_dataset['class']

positive_indices = [i for i, label in enumerate(labels) if label == 1]
negative_indices = [i for i, label in enumerate(labels) if label == 0]
neutral_indices = [i for i, label in enumerate(labels) if label == 2]

train_pos_indices, valid_pos_indices = train_test_split(positive_indices, test_size=0.2, random_state=66)
train_neg_indices, valid_neg_indices = train_test_split(negative_indices, test_size=0.2, random_state=66)
train_neu_indices, valid_neu_indices = train_test_split(neutral_indices, test_size=0.2, random_state=66)

valid_pos_indices, test_pos_indices = train_test_split(valid_pos_indices, test_size=0.5, random_state=66)
valid_neg_indices, test_neg_indices= train_test_split(valid_neg_indices, test_size=0.5, random_state=66)
valid_neu_indices, test_neu_indices= train_test_split(valid_neu_indices, test_size=0.5, random_state=66)


valid_indices = valid_pos_indices + valid_neg_indices + valid_neu_indices
train_indices = train_pos_indices + train_neg_indices + train_neu_indices
test_indices = test_pos_indices + test_neg_indices + test_neu_indices

np.random.shuffle(train_indices)

new_train_dataset = train_dataset.select(train_indices)
eval_dataset = train_dataset.select(valid_indices)
test_dataset = train_dataset.select(test_indices)

# Define sentiment words
positive_words = ['offensive"']
neutral_words = ['neutral']
negative_words = ['hateful']

positive_word_ids = tokenizer.convert_tokens_to_ids(positive_words)
neutral_word_ids = tokenizer.convert_tokens_to_ids(neutral_words)
negative_word_ids = tokenizer.convert_tokens_to_ids(negative_words)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def collate_fn(batch):
    input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item['class'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}



def calculate_sentiment_scores(batch):
  input_ids = torch.tensor(batch['input_ids']).to(device)
  attention_mask = torch.tensor(batch['attention_mask']).to(device)
  with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = outputs.logits
  sentiment_scores = []
  for i in range(predictions.shape[0]):
    mask_token_indices = (input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if mask_token_indices[0].size(0) == 0:
      continue
    masked_index = mask_token_indices[0].item()

    logits_at_masked_position = predictions[i, masked_index]

    positive_logits = logits_at_masked_position[positive_word_ids]
    neutral_logits = logits_at_masked_position[neutral_word_ids]
    negative_logits = logits_at_masked_position[negative_word_ids]

    combined_logits = torch.cat((positive_logits, neutral_logits, negative_logits))
    probabilities = torch.softmax(combined_logits, 0)

    positive_prob_sum = torch.sum(probabilities[:len(positive_logits)])
    neutral_prob_sum = torch.sum(probabilities[len(positive_logits):len(positive_logits) + len(neutral_logits)])
    negative_prob_sum = torch.sum(probabilities[len(positive_logits) + len(neutral_logits):])

    total_prob_sum = (positive_prob_sum + neutral_prob_sum + negative_prob_sum)

    # Normalize probabilities

    positive_prob_normalized = positive_prob_sum / total_prob_sum
    neutral_prob_normalized = neutral_prob_sum / total_prob_sum
    negative_prob_normalized = negative_prob_sum / total_prob_sum


  #   # Calculate sentiment score
  #   sentiment_score = torch.abs(positive_prob_normalized - negative_prob_normalized).item()
  #   sentiment_scores.append(sentiment_score)
  # return sentiment_scores
    # Collect all normalized probabilities
    normalized_probs = torch.tensor([positive_prob_normalized, neutral_prob_normalized, negative_prob_normalized])

    # Get the top two highest probabilities
    top_probs, _ = torch.topk(normalized_probs, 2)

    # Calculate the sentiment score
    sentiment_score = torch.abs(top_probs[0] - top_probs[1]).item()
    sentiment_scores.append(sentiment_score)

  return sentiment_scores

def calculate_initial_scores(dataset, tokenizer, model, device):
    initial_scores = []

    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Calculating scores")):
            batch_scores = calculate_sentiment_scores(batch)
            for i, score in enumerate(batch_scores):
                initial_scores.append((idx * 16 + i, score))  # Include the index along with the score

    return initial_scores

def sort_dataset_by_token_length(dataset, tokenizer):
    lengths = []
    for i in range(len(dataset)):
        input_ids = dataset[i]['input_ids']
        lengths.append((i, len(input_ids)))
    sorted_indices = [idx for idx, length in sorted(lengths, key=lambda x: x[1])]
    return dataset.select(sorted_indices)


sorted_train_dataset = sort_dataset_by_token_length(new_train_dataset, tokenizer)

sorted_train_dataset = sorted_train_dataset.select(range(64))

print("\n🔍 Token lengths of top 5 sorted samples:")
for i in range(5):
    print(f"[{i}] Token length: {len(sorted_train_dataset[i]['input_ids'])}")

train_loader = DataLoader(sorted_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


eval_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=collate_fn)


optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 5
batch_size = 16
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


scores = calculate_initial_scores(new_train_dataset, tokenizer, model, device)
score_values = [score for _, score in scores]


logging.basicConfig(filename=f'{method}_{data}_{seed}.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

save_path = f'datathesis_{data}'
os.makedirs(save_path, exist_ok=True)
checkpoint_path = os.path.join(save_path, f"checkpoint_{method}_{data}_{seed}.pth")

best_val_accuracy = 0
best_model_path = os.path.join(save_path, f'best_model_{method}_{data}_{seed}.pth')

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs, labels = batch['input_ids'], batch['labels']
        outputs = model(input_ids=inputs, attention_mask=batch['attention_mask'], labels=inputs)
        logits = outputs.logits

        mask_token_index = (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        scores = torch.zeros(size=(len(labels), len(label_words)), device=device)

        for i, idx in enumerate(mask_token_index):
            logits_at_mask = logits[i, idx, :]
            class_logits = []

            for label, word_ids in verbalizer_ids.items():
                logits_for_words = logits_at_mask[word_ids]
                class_logits.append(logits_for_words)

            all_logits = torch.cat(class_logits, dim=0)
            probabilities = F.softmax(all_logits, dim=0)

            start_index = 0
            for label, word_ids in verbalizer_ids.items():
                end_index = start_index + len(word_ids)
                scores[i, label] = probabilities[start_index:end_index].sum()
                start_index = end_index

        log_scores = torch.log(scores + 1e-10)
        loss = F.nll_loss(log_scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted_labels = torch.argmax(scores, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples
    print(f"Epoch {epoch+1}: Avg loss = {avg_loss}, Train Accuracy = {train_accuracy}")
    logger.info(f"Epoch {epoch+1}: Avg loss = {avg_loss}, Train Accuracy = {train_accuracy}")



    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()}, checkpoint_path)


    model.eval()
    eval_loss = 0
    eval_correct = 0
    eval_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs, labels = batch['input_ids'], batch['labels']
            outputs = model(input_ids=inputs, attention_mask=batch['attention_mask'], labels=inputs)
            logits = outputs.logits

            mask_token_index = (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            scores = torch.zeros(size=(len(labels), len(label_words)), device=device)

            for i, idx in enumerate(mask_token_index):
                logits_at_mask = logits[i, idx, :]
                class_logits = []

                for label, word_ids in verbalizer_ids.items():
                    logits_for_words = logits_at_mask[word_ids]
                    class_logits.append(logits_for_words)

                all_logits = torch.cat(class_logits, dim=0)
                probabilities = F.softmax(all_logits, dim=0)

                start_index = 0
                for label, word_ids in verbalizer_ids.items():
                    end_index = start_index + len(word_ids)
                    scores[i, label] = probabilities[start_index:end_index].sum()
                    start_index = end_index

            log_scores = torch.log(scores + 1e-10)
            loss = F.nll_loss(log_scores, labels)

            eval_loss += loss.item()
            predicted_labels = torch.argmax(scores, dim=1)
            eval_correct += (predicted_labels == labels).sum().item()
            eval_samples += len(labels)

    avg_eval_loss = eval_loss / len(eval_loader)
    eval_accuracy = eval_correct / eval_samples
    print(f"Epoch {epoch+1}: Avg Eval Loss = {avg_eval_loss}, Eval Accuracy = {eval_accuracy}")
    logger.info(f"Epoch {epoch+1}: Avg Eval Loss = {avg_eval_loss}, Eval Accuracy = {eval_accuracy}")


     # Save the best model
    if eval_accuracy > best_val_accuracy:
        best_val_accuracy = eval_accuracy
        torch.save(model.state_dict(), best_model_path)


print("Training complete.")

# Create data loaders for training and validation
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Load the best model
model.load_state_dict(torch.load(best_model_path))

model.eval()

test_loss = 0
test_correct = 0
test_samples = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs, labels = batch['input_ids'], batch['labels']
        outputs = model(input_ids=inputs, attention_mask=batch['attention_mask'], labels=inputs)
        logits = outputs.logits

        mask_token_index = (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        scores = torch.zeros(size=(len(labels), len(label_words)), device=device)

        for i, idx in enumerate(mask_token_index):
            logits_at_mask = logits[i, idx, :]
            class_logits = []

            for label, word_ids in verbalizer_ids.items():
                logits_for_words = logits_at_mask[word_ids]
                class_logits.append(logits_for_words)

            all_logits = torch.cat(class_logits, dim=0)
            probabilities = F.softmax(all_logits, dim=0)

            start_index = 0
            for label, word_ids in verbalizer_ids.items():
                end_index = start_index + len(word_ids)
                scores[i, label] = probabilities[start_index:end_index].sum()
                start_index = end_index

        log_scores = torch.log(scores + 1e-10)
        loss = F.nll_loss(log_scores, labels)

        test_loss += loss.item()
        predicted_labels = torch.argmax(scores, dim=1)
        test_correct += (predicted_labels == labels).sum().item()
        test_samples += len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted_labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = test_correct / test_samples

# Calculate precision, recall, f1-score
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

print(f"Avg Test Loss = {avg_test_loss}, Test Accuracy = {test_accuracy}")
print(f"Precision = {precision}, Recall = {recall}, F1 Score = {f1}")
logger.info(f"Avg Test Loss = {avg_test_loss}, Test Accuracy = {test_accuracy}")


