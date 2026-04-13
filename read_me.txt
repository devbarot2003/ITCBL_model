7104_PROJECT/
│
├── data/
│     └── csic_database.csv # Dataset (CSIC 2010)
│
├── models/
│		├── ITCBLModel.py 				# Hybrid model architecture
│		├── tokenizer.pkl 				# Saved tokenizer
│ 		├── sqli_model.pth 				# Trained model
│ 		├── sqli_model_attention.pth 	# Model with attention
│ 		└── sqli_model_no_attention.pth
│
├── preprocessing/
│ 		├── prepare_data.py 			# Data preparation pipeline
│ 		├── dataset.py 					# Dataset loader
│ 		├── tokenizer.py 				# Tokenizer logic
│ 		└── text_preprocessing.py 		# Cleaning & normalization
│
├── training/
│ 		├── train_model.py 				# Model training
│ 		├── test_model.py 				# Testing pipeline
│ 		├── sql_project.py 				# Main training script
│ 		└── check_labels.py 			# Label validation
│
├── evaluation/
│ 		├── adversarial_test.py 		# Robustness testing
│		├── adversarial_manual.py 		# Manual adversarial inputs
│		├── compare_models.py 			# Model comparison
│ 		├── context_aware_test.py 		# Context-based evaluation
│		├── explain_prediction.py 		# Prediction explanation
│		├── latency_test.py 			# Latency measurement
│ 		└── test_session_tracker.py 	# Session tracking tests
│
├── security/
│ 		└── session_tracker.py 			# Session-level tracking logic
│
└── README.md

---------------------------------------------------------------------------

uploading on GitHub
git clone <ITCBL_model> 
and then using GitHub desktop performing the operation of commit and push to origin

---------------------------------------------------------------------------

Model Architecture
1. Embedding Layer - Converts tokens to vectors
2. TextCNN - Extracts local patterns (keywords, operators)
3. Bi-LSTM - Captures sequential dependencies
4. Attention Layer - Focuses on important features
5. Dense + Sigmoid - Final classification

---------------------------------------------------------------------------

Dataset
CSIC 2010 Web Application Dataset
Stored in: data/csic_database.csv
Contains:
	Normal HTTP requests
	SQL injection attacks

---------------------------------------------------------------------------

Environment - Python 3.11.x, Visual Studio Code and Anaconda Prompt

Libraries

Deep Learning
	torch – Model building and training (PyTorch)
	torchvision – Utilities for PyTorch
Data Processing
	pandas, numpy, scikit-learn, regex, pickle

Scikit-Learn Metrics (Model Evaluation)
accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

System libraries - os, sys
Performance MEasurement - time

---------------------------------------------------------------------------

Results:
Accuracy: ~98%
High recall for malicious queries (~0.99)
AUC: ~0.97
Latency: ~1.77 ms per query
Robustness: ~95% accuracy on modified queries

---------------------------------------------------------------------------

Usage
to preprocess data - preprocessing/prepare_data.py
to Train the model - python training/train_model.py
to Test the model - python training/test_model.py
to Run adversarial evaluation - python evaluation/adversarial_test.py
to Measure latency - python evaluation/latency_test.py