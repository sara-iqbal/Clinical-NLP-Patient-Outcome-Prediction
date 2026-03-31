# Clinical NLP — Patient Outcome Prediction

This repository contains a specialized NLP pipeline designed to classify medical text—such as clinical notes or research abstracts—into predicted patient outcomes. Using **DistilBERT**, the project demonstrates how transformer-based models can be fine-tuned to interpret complex medical narratives and provide structured insights.

---

## Project Overview

The core objective is to automate the classification of clinical documentation into three distinct categories:
* **Positive Outcome:** Indicators of recovery, treatment success, or resolution of symptoms.
* **Negative Outcome:** Indicators of deterioration, poor prognosis, or treatment failure.
* **Uncertain:** Cases where the response is partial, mixed, or requires further monitoring.

The notebook is designed with a **robust fallback mechanism**: it attempts to use the `PubMedQA` dataset (BigBio) for real-world medical abstract analysis, but includes a synthetic clinical note generator to ensure the pipeline is fully functional even in offline or restricted environments.

---

## Technical Stack

* **Model:** `distilbert-base-uncased` (chosen for its balance of performance and computational efficiency).
* **Frameworks:** Hugging Face `transformers`, `datasets`, `PyTorch`.
* **Preprocessing:** Custom tokenization with truncation and padding handled by `DataCollatorWithPadding`.
* **Evaluation:** Scikit-learn for F1-macro, weighted F1, and confusion matrix visualization.

---

##  Pipeline Architecture

### 1. Data Preparation
The script implements a dynamic data loader:
* **Primary:** Loads `pubmed_qa` and maps decisions (yes/no/maybe) to outcome labels.
* **Fallback:** Generates 3,000 synthetic clinical records using medical templates involving common symptoms (e.g., chest pain, hypertension) and treatments (e.g., ACE inhibitors, insulin).

### 2. Fine-Tuning
The model is trained using the Hugging Face `Trainer` API with the following configuration:
* **Epochs:** 3
* **Batch Size:** 16 (Train) / 32 (Eval)
* **Optimizer:** AdamW (default in `TrainingArguments`)
* **Metrics:** Accuracy and Macro-averaged F1 score.

### 3. Evaluation & Visualization
Post-training, the model generates a comprehensive performance report including:
* **Classification Report:** Precision, Recall, and F1 per class.
* **Confusion Matrix:** A heatmap visualizing where the model might be misclassifying specific outcomes.
* **Per-Class F1 Bar Chart:** Highlighting the model's reliability across the three different labels.

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. You will need a GPU (CUDA) for optimal training speeds, though the script automatically falls back to CPU.

### Installation
```bash
pip install transformers datasets scikit-learn torch pandas numpy matplotlib seaborn
```

### Execution
Run the notebook or export it to a Python script:
1.  **Step 1:** Imports and environment setup.
2.  **Step 2:** Dataset loading and label mapping.
3.  **Step 3:** Tokenization and DistilBERT fine-tuning.
4.  **Step 4:** Evaluation and export of `clinical_nlp_data.json` for dashboarding.

---

## Outputs

Upon completion, the project generates:
* `clinical_nlp_results.png`: A high-resolution visualization of model performance.
* `clinical_nlp_data.json`: A structured summary of metrics (Accuracy, F1, Confusion Matrix) for use in external reports or web dashboards.

---
**Author:** Sara Iqbal | MSc Data Science
