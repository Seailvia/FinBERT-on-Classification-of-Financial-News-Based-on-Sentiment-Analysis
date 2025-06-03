import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

def calculate_metrics(file_path):
    true_labels = []
    pred_labels = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            sentence = row['sentence'].strip()
            prediction = int(row['prediction'].strip())  # 0, 1, or 2
            
            # Extract the true label (after @)
            if '@' in sentence:
                at_part = sentence.split('@')[-1]
                tag = at_part.split()[0].lower().strip()
            else:
                continue  # Skip if no @ tag
                
            # Map prediction to label
            if prediction == 0:
                pred_tag = 'negative'
            elif prediction == 1:
                pred_tag = 'positive'
            elif prediction == 2:
                pred_tag = 'neutral'
            else:
                continue  # Invalid prediction, skip
                
            true_labels.append(tag)
            pred_labels.append(pred_tag)
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    total = len(true_labels)
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    # Generate confusion matrix
    labels = ['negative', 'positive', 'neutral']  # Ensure consistent order
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    # Calculate precision, recall, f1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=labels, average=None)
    
    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average='weighted')
    
    # Create a classification report dictionary
    report = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion_matrix': cm,
        'labels': labels,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    }
    
    return report

file_path = '.\output\predictions.csv'
report = calculate_metrics(file_path)

# Print accuracy
print(f"Accuracy: {report['accuracy']:.2f}% ({report['correct']}/{report['total']})")

# Print classification report
print("\nClassification Report:")
print("{:<15} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "F1-Score"))
for i, label in enumerate(report['labels']):
    print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(
        label, report['precision'][i], report['recall'][i], report['f1'][i]))
print("\nMacro Avg:    {:<10.2f} {:<10.2f} {:<10.2f}".format(
    report['macro_avg']['precision'], report['macro_avg']['recall'], report['macro_avg']['f1']))
print("Weighted Avg: {:<10.2f} {:<10.2f} {:<10.2f}".format(
    report['weighted_avg']['precision'], report['weighted_avg']['recall'], report['weighted_avg']['f1']))

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
            xticklabels=report['labels'], yticklabels=report['labels'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()