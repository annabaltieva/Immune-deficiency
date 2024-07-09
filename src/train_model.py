from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from datasets import Dataset
import pandas as pd
import torch

# Load the dataset using pandas
df = pd.read_csv('../data/therapy_classification.csv')

# Ensure labels are integers
df['output'] = df['output'].apply(lambda x: 1 if x == 'YES' else 0)

# Split the dataset into train and eval sets
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

# Convert the dataframe to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['input'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Ensure 'labels' column is present
train_dataset = train_dataset.rename_column("output", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

eval_dataset = eval_dataset.rename_column("output", "labels")
eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Debug: Print a sample from the tokenized dataset to verify labels
print(train_dataset[0])
print(eval_dataset[0])

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='../results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../logs',
    logging_steps=10,
    save_total_limit=2,  # Limit the total amount of checkpoints
    save_steps=10,  # Save checkpoint every 10 steps
)

# Custom Trainer class to override the compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize the Trainer with the custom compute_loss method
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model('../results/model')
tokenizer.save_pretrained('../results/model')