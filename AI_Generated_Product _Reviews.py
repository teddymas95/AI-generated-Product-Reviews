import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset

# Load the dataset
def load_data(mal_file_path, tam_file_path):
    try:
        mal_data = pd.read_csv(mal_file_path)
        tam_data = pd.read_csv(tam_file_path)
        return mal_data, tam_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit(1)

def preprocess_data(df, tokenizer, text_column, label_column, max_length=64):
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=max_length)
    df[label_column] = df[label_column].astype('category')
    categories = df[label_column].cat.categories.tolist()
    df[label_column] = df[label_column].cat.codes
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, categories

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()
    return cm

# Model comparison table
def create_comparison_table(models_metrics, model_names):
    comparison_df = pd.DataFrame(models_metrics, index=model_names, 
                                 columns=['Precision', 'Recall', 'F1-Score', 'Accuracy'])
    return comparison_df

# Error analysis helper
def error_analysis(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    errors = []
    for i in range(len(labels)):
        fn = cm[i, :].sum() - cm[i, i]  # False Negatives
        fp = cm[:, i].sum() - cm[i, i]  # False Positives
        errors.append({
            'Class': labels[i],
            'False Positives': fp,
            'False Negatives': fn
        })
    error_df = pd.DataFrame(errors)
    print(f"\nError Analysis for {model_name}:")
    print(error_df)
    return error_df

# Load data
mal_data, tam_data = load_data('AI DATA/mal_training_data_hum_ai.csv', 'AI DATA/tam_training_data_hum_ai.csv')
mal_data.rename(columns={'DATA': 'review', 'LABEL': 'label', 'ID': 'id'}, inplace=True)
tam_data.rename(columns={'DATA': 'review', 'LABEL': 'label', 'ID': 'id'}, inplace=True)

# Model names
model_names = ["xlm-roberta-base", "distilbert-base-uncased", "bert-base-multilingual-cased"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./AI Gen",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2
)

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

# Function to train and evaluate a model
def train_and_evaluate(model_name, train_dataset, test_dataset, num_labels, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    predictions = trainer.predict(test_dataset).predictions
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_dataset['label']
    return pred_classes, true_classes, trainer

# Process for each language and model
for lang, data in [("Malayalam", mal_data), ("Tamil", tam_data)]:
    print(f"\nProcessing {lang} Data\n{'='*20}")
    
    # Get labels for the current language
    tokenizer = AutoTokenizer.from_pretrained(model_names[0])  # Temporary tokenizer to get labels
    tokenized_dataset, labels = preprocess_data(data, tokenizer, 'review', 'label')
    
    # Metrics storage
    models_metrics = []
    confusion_matrices = []

    for model_name in model_names:
        print(f"\nTraining {model_name} for {lang}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_dataset, _ = preprocess_data(data, tokenizer, 'review', 'label')  # Re-tokenize for each model
        splits = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        pred_classes, true_classes, trainer = train_and_evaluate(model_name, splits['train'], splits['test'], len(labels), tokenizer)
        
        # Confusion Matrix
        cm = plot_confusion_matrix(true_classes, pred_classes, labels, f"{lang} Confusion Matrix - {model_name}")
        confusion_matrices.append(cm)
        
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(true_classes, pred_classes, average='macro')
        accuracy = (pred_classes == true_classes).mean()
        models_metrics.append([precision, recall, f1, accuracy])
        
        # Error Analysis
        error_analysis(true_classes, pred_classes, labels, model_name)
        
        # Classification Report
        print(f"\n{lang} Classification Report for {model_name}:")
        print(classification_report(true_classes, pred_classes, target_names=labels))

    # Comparison Table
    comparison_table = create_comparison_table(models_metrics, model_names)
    print(f"\n{lang} Model Comparison Table:")
    print(comparison_table)

    # Plot comparison graph
    comparison_table.plot(kind='bar', figsize=(10, 6))
    plt.title(f"{lang} Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.show()