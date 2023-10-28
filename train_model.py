import random
from collections import defaultdict, Counter

def parse_conll(filepath, flag=True):
    lines = open(filepath).readlines()
    tokens, labels, sentence, sentence_label = [], [], [], []
    ct = 0
    for line in lines:
        line = line.strip()

        if line == "":
            if len(sentence) > 0:
                counts = Counter(sentence_label)
                if len(sentence)-counts['O'] <= 0 and flag:
                    ct += 1
                else:
                    tokens.append(sentence)
                    labels.append(sentence_label)
            sentence = []
            sentence_label = []
            continue

        sentence.append(line.split()[0])
        sentence_label.append(line.split()[-1])

    if len(sentence) > 0:
        tokens.append(sentence)
        labels.append(sentence_label)
    print("Ignored ", ct, "number of sentences")
    return tokens, labels

train_tokens, train_labels = parse_conll("train.conll")
test_tokens, test_labels = parse_conll("test.conll", flag=False)
print("Train and test size before splitting paragraphs: ", len(train_tokens), len(test_tokens))

def split_sentence(tokens):
  sentences, sentence = [], []
  N = len(tokens)

  if N < 350:
    return [tokens]

  for tok in tokens:
    if tok == ".":
      sentence.append(".")
      sentences.append(sentence)
      sentence = []
    else:
      sentence.append(tok)

  if len(sentence) > 0:
    sentences.append(sentence)

  splits, paragraph = [], []

  len_so_far = 0
  for sentence in sentences:
    n = len(sentence)

    if len_so_far + n >= 350:
      splits.append(paragraph)
      paragraph = []
      len_so_far = 0

    paragraph += sentence
    len_so_far += len(sentence)

  if len(paragraph) > 0:
    splits.append(paragraph)
  return splits

filtered_train, filtered_test = [], []
for para in train_tokens:
  splits = split_sentence(para)
  for val in splits:
    filtered_train.append(val)

for sent in test_tokens:
  splits = split_sentence(sent)
  for val in splits:
    filtered_test.append(val)

print("Filtered train and test size: ", len(filtered_train), len(filtered_test))

label_names = set()

from collections import defaultdict
lab_count = defaultdict(int)
for lab in train_labels:
  for lb in lab:
    label_names.add(lb)
    lab_count[lb] += 1
label_names = list(label_names)
lab2ind = {lab:i for i,lab in enumerate(label_names)}
train_labels = [[lab2ind[lab] for lab in inner] for inner in train_labels]
test_labels = [[lab2ind[lab] for lab in inner] for inner in test_labels]

new_dataset = {"train": [], "test": []}
i = 0
for t, l in zip(train_tokens, train_labels):
  temp = {}
  temp['id'] = str(i)
  i+=1
  temp['tokens'] = t
  temp['ner_tags'] = l
  new_dataset['train'].append(temp)

for t, l in zip(test_tokens, test_labels):
  temp = {}
  temp['id'] = str(i)
  i+=1
  temp['tokens'] = t
  temp['ner_tags'] = l
  new_dataset['test'].append(temp)

from transformers import AutoTokenizer

model_checkpoint = "KISTI-AI/scideberta-cs"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    new_labels = align_labels_with_tokens(examples['ner_tags'], tokenized_inputs.word_ids())

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

train_tokenized_datasets = []
for sample in new_dataset['train']:
  train_tokenized_datasets.append(tokenize_and_align_labels(sample))

test_tokenized_datasets = []
for sample in new_dataset['test']:
  test_tokenized_datasets.append(tokenize_and_align_labels(sample))

print("First tokenized train and test sample: ", train_tokenized_datasets[0], test_tokenized_datasets[0])

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

import evaluate
metric = evaluate.load("seqeval")

import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return all_metrics

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
from transformers import AutoModelForTokenClassification

# IBM
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# SECOND TIME ON OUR DATA

# model = AutoModelForTokenClassification.from_pretrained(
#     "./scibert-ibm/checkpoint-900",
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True
# )

import accelerate
import transformers
from transformers import TrainingArguments

args = TrainingArguments(
    "scibert-nlp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=7,
    weight_decay=0.01,
    push_to_hub=False,
    use_cpu=False,
    auto_find_batch_size=True
)

# Rescaling the loss function

N = sum(lab_count.values())
ratio = [N/lab_count.get(id2label[key], N) for key in id2label]

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(ratio, device=model.device))
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

trainer.train()