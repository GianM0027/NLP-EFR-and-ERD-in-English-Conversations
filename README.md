# Emotion Discovery and Reasoning its Flip in Conversation (EDiReF)

### Course Project - Natural Language Processing

This repository contains the code and documentation for the standard course project of the Natural Language Processing (NLP) course at the University of Bologna, conducted by Professor [Paolo Torroni](https://www.unibo.it/sitoweb/p.torroni) at the University of Bologna.

## Overview
The assignment focuses on the [EDiReF challenge (subtask iii)](https://lcs2.in/SemEval2024-EDiReF/). The primary objective was to address a double multi-label classification task by creating a BERT-based model capable of performing:
* ERC: Given a dialogue, ERC aims to assign an emotion to each utterance from a predefined set of possible emotions.
* EFR: Given a dialogue, EFR aims to identify the trigger utterance(s) for an emotion-flip in a multi-party conversation dialogue.


## Requirements
Ensure you have the necessary dependencies by checking the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
## Main Notebook
The main notebook (`main_notebook.ipynb`) serves as the central hub for this project. By executing this notebook, you can perform the following tasks:

- **Data Preparation and Exploration:** the dataset is split, cleaned and analyzed in order to find significant information.
- **Model Creation:** Implementation, training, and evaluation of the bert-based models.
- **Error Analysis:** error analysis of the bert-based models, comparation of the performance across the models.

Feel free to explore and customize the main notebook to experiment with different configurations and settings.


## drTorch Framework

The "drTorch" folder contains a framework developed from scratch for creating neural network models using PyTorch. This framework was created during the first assignment of the course and has been customized, 
extended, and adapted throughout the projects completed during this course.

## Models Implementation

The `models` folder contains the implementation of neural models, including:
* **Random Uniform Classifier:** An individual classifier per category.
* **Majority Classifier:** An individual classifier per category.
* **BertOne:** A BERT-based classifier for emotions and triggers.


## Authors:
For any questions or assistance, feel free to contact:
- [Mauro Dore](mauro.dore@studio.unibo.it)
- [Giacomo Gaiani](giacomo.gaiani@studio.unibo.it)
- [Gian Mario Marongiu](gianmario.marongiu@studio.unibo.it)
- [Riccardo Murgia ](riccardo.murgia2@studio.unibo.it)


