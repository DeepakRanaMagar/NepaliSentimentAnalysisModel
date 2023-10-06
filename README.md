---
license: apache-2.0
language:
- ne
library_name: transformers
pipeline_tag: text-classification
---

# Sentiment Analysis with BERT: Nepali Sentiment Dataset

This repository contains code for training and evaluating a sentiment analysis model using the BERT (Bidirectional Encoder Representations from Transformers) model on the Nepali Sentiment Dataset. The model achieves an accuracy of 99.75% on the test dataset.

This Model is Hosted on huggingface, You can access this model with the link below:
https://huggingface.co/dpkrm/NepaliSentimentAnalysis

## Dataset

The dataset used for training and testing the sentiment analysis model is a balanced dataset in CSV format. The dataset is loaded using the `pandas` library. The training dataset consists of 2084 balanced data, and the test dataset consists of 2001 balanced data. Label 0 = Negative, Label 1 = Positive, Label 2 = Neutral 


## Model

The BERT model is used for sequence classification and is loaded from the `bert-base-multilingual-uncased` pre-trained model. The model is initialized with `num_labels=3` since we have three sentiment classes: positive, negative, and neutral.

## Preprocessing

The dataset is preprocessed using the `NepaliSentimentDataset` class. The class takes the texts, labels, tokenizer, and maximum sequence length as inputs. The texts are preprocessed using regular expressions to remove special characters, usernames, and extra whitespace. The `tokenizer` from the Hugging Face `transformers` library is used to tokenize the texts and convert them into input IDs and attention masks. The preprocessed data is returned as a dictionary with the input IDs, attention masks, and labels.

## Training

The model is trained using the `train_model` function. The function takes the model, train dataloader, and test dataloader as inputs. The model is trained for 10 epochs with an early stopping mechanism. The AdamW optimizer is used with a learning rate of 2e-5 and epsilon value of 1e-8. The function also includes additional connection layers before the classification layer of the BERT model. After each epoch, the model is evaluated on the test dataset.

## Training Progress and Evaluation Metrics
This section provides insights into the training progress of the sentiment analysis model and includes graphs showing the loss values and accuracy values throughout the training process.

# Loss Value Graph
The graph below displays the training progress by showing the variation in the loss values across different epochs. It helps visualize the convergence of the model during training.

Loss Value Graph

# Accuracy Value Graph
The following graph illustrates the accuracy values achieved by the model during the training process. It presents a clear picture of how the model's performance improves over time.

Accuracy Value Graph

These graphs provide a visual representation of the training progress and performance of the sentiment analysis model, allowing for better understanding and analysis of the results.

## Results

After training, the trained model achieves an accuracy of 99.75% on the test dataset.

## Saving the Model
The trained model and tokenizer are saved using the `save_pretrained` function from the Hugging Face `transformers` library. The model and tokenizer are saved in the directory 

---

**Note:** The code provided is a simplified version for demonstration purposes. Additional error handling, logging, and hyperparameter tuning can be added for production use.
