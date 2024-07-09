from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the trained model and tokenizer
model_path = '../results/model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load the evaluation dataset
eval_df = pd.read_csv('../data/therapy_classification.csv')

# Ensure labels are integers
eval_df['output'] = eval_df['output'].apply(lambda x: 1 if x == 'YES' else 0)

# Convert the dataframe to a Dataset object
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['input'], padding='max_length', truncation=True)

eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Ensure 'labels' column is present
eval_dataset = eval_dataset.rename_column("output", "labels")
eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define the compute metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Initialize the Trainer
training_args = TrainingArguments(
    output_dir='../results',
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Evaluate the model
eval_result = trainer.evaluate()

print(f"Evaluation results: {eval_result}")

# Additional detailed metrics
predictions = trainer.predict(eval_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Generate a detailed classification report
from sklearn.metrics import classification_report
report = classification_report(labels, preds, target_names=['NO', 'YES'])
print(report)

# Generate a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("../results/confusion_matrix.png")
plt.show()