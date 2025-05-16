import csv
import ast 
import re
import os
import ftfy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict
from typing import List, Set
from ast import literal_eval
from tqdm import tqdm
tqdm.pandas()


"""


"""

def get_tweets_dataset():
    """
    Load the dataset & clean the text
    return a DataFrame with 'text' and 'label' columns
    """
    # tweets = pd.read_csv('data/tweets.csv', encoding="utf-8-sig")
    tweets = pd.read_csv('data/tweets_with_synonyms_matching.csv', encoding="utf-8-sig")

    # all special characters were displayed weird, ftfy is a library that fixes text encoding issues
    def fix_text_cell(cell):
        if isinstance(cell, str):
            return ftfy.fix_text(cell)
        return cell


    def safe_convert(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception as e:
                return []
        return x

    # Apply the fix to all text (object) columns in robust_data
    for col in tweets.select_dtypes(include=["object"]).columns:
        tweets[col] = tweets[col].apply(fix_text_cell)
        #tweets[col] = tweets[col].apply(safe_convert)
    return tweets

def extract_responses(tweets, directory):
    extract = []
    for i, row in tqdm(tweets.iterrows(), total=tweets.shape[0]):
        with open(f"data/{directory}/{i}.json", 'r', encoding='utf-8-sig') as f:
            content = f.read()
            extract.append(content)
    return extract

def extract_JSON_labels_and_explanations(tweets, directory: str):
    # a lot of responses from the LLM's  are not perfect JSON
    # so we need to extract the JSON block from the response
    # and then parse it
    JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)

    labels = []
    explanations = []
    for i, row in tqdm(tweets.iterrows(), total=len(tweets)):
        with open(f"data/{directory}/{i}.json", "r", encoding="utf‑8‑sig") as f:
            raw = f.read().strip()
        # Find JSON block
        m = JSON_BLOCK.search(raw)
        if not m:
            print(f"Error: No JSON found in '{directory}/{i}'")
            labels.append("NO_JSON_FOUND")
            explanations.append("Could not locate JSON braces")
            continue
        json_str = m.group(0)
        try:
            data = json.loads(json_str)
            labels.append(data.get("label", "MISSING_LABEL"))
            explanations.append(data.get("explanation", "MISSING_EXPLANATION"))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{directory}/{i}': {e}")
            labels.append("JSON_ERROR")
            explanations.append(f"Decode error: {e}")
    return labels, explanations


def extract_T_F_labels(tweets, directory: str):
    THINK_RE   = re.compile(r"<think>(.*?)</think>", re.I | re.S)
    LABEL_RE   = re.compile(r"\b([TF])\b")        # lone T or F
    labels = []
    for i, row in tqdm(tweets.iterrows(), total=len(tweets)):
        with open(f"data/{directory}/{i}.json", "r", encoding="utf‑8‑sig") as f:
            raw = f.read().strip()
        # Find <think> block
        m_think = THINK_RE.search(raw)

        # remove think block before searching for label
        if m_think:
            cleaned = raw[:m_think.start()] + raw[m_think.end():].strip()
        else:
            cleaned = raw.strip()

        m_label = LABEL_RE.search(cleaned)
        if m_label:
            label = m_label.group(1).strip()
            if label != "T" and label != "F":
                print(f"Error: Invalid label '{label}' in '{directory}/{i}'")
                labels.append("INVALID_LABEL")
            else:
                labels.append(label)
        else:
            print(f"Error: No label found '{raw}' in '{directory}/{i}'")
            labels.append("NO_LABEL_FOUND")
    return labels


def extract_drug_labels(tweets: pd.DataFrame, directory: str) -> List[str]:
    # Regex to capture a JSON array block (including nested objects)
    ARRAY_BLOCK = re.compile(r"\[\s*(?:\{[\s\S]*?\}\s*,?\s*)+\]", re.DOTALL)
    THINK_RE   = re.compile(r"<think>(.*?)</think>", re.I | re.S)
    drug_labels = []
    for i, row in tqdm(tweets.iterrows(), total=len(tweets)):
        unique_terms: Set[str] = set()
        with open(f"data/{directory}/{i}.json", "r", encoding="utf‑8‑sig") as f:
            raw = f.read().strip()
        # remove <think> block
        m_think = THINK_RE.search(raw)
        if m_think:
            cleaned = raw[:m_think.start()] + raw[m_think.end():].strip()
        else:
            cleaned = raw.strip()
        # Find JSON block
        m = ARRAY_BLOCK.search(cleaned)
        if not m:
            if cleaned == "[]" or cleaned == "```json\n[]\n```":
                drug_labels.append('')
                continue
            else:
                print(f"Error: No JSON found in '{directory}/{i}   {raw}'")
                drug_labels.append("NO_JSON_FOUND")
                continue
        try:
            entries = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            print(f"Error2: No JSON found in '{directory}/{i}   {raw}'")
            drug_labels.append("Error2")
            continue
        if isinstance(entries, list):
            for obj in entries:
                term = obj.get("index_term")
                if isinstance(term, str):
                    unique_terms.add(term)
            drug_labels.append(list(unique_terms))
        else:
            print(f"Error3: in '{directory}/{i}   {raw}'")
            drug_labels.append("Error3")
    return drug_labels

def extract_RAG_drug_labels(tweets: pd.DataFrame, directory: str) -> List[str]:
    # Regex to capture a JSON array block (including nested objects)
    ARRAY_BLOCK = re.compile(r"\[\s*(?:\{[\s\S]*?\}\s*,?\s*)+\]", re.DOTALL)
    THINK_RE   = re.compile(r"<think>(.*?)</think>", re.I | re.S)
    drug_labels = []
    for i, row in tqdm(tweets.iterrows()):
        unique_terms: Set[str] = set()
        path = Path(f"data/{directory}/{i}.json")
        if not path.exists():
            drug_labels.append("NO_RAG_CHUNKS")
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read().strip()
        # remove <think> block
        m_think = THINK_RE.search(raw)
        if m_think:
            cleaned = raw[:m_think.start()] + raw[m_think.end():].strip()
        else:
            cleaned = raw.strip()
        # Find JSON block
        m = ARRAY_BLOCK.search(cleaned)
        if not m:
            if cleaned == "[]" or cleaned == "```json\n[]\n```":
                drug_labels.append('')
                continue
            else:
                print(f"Error: No JSON found in '{directory}/{i}   {raw}'")
                drug_labels.append('')
                continue
        try:
            entries = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            print(f"Error2: No JSON found in '{directory}/{i}   {raw}'")
            drug_labels.append("Error2")
            continue
        if isinstance(entries, list):
            for obj in entries:
                term = obj.get("index_term")
                if isinstance(term, str):
                    unique_terms.add(term)
            drug_labels.append(list(unique_terms))
        else:
            print(f"Error3: in '{directory}/{i}   {raw}'")
            drug_labels.append("Error3")
    print(f"processed'{len(drug_labels)}'")
    return drug_labels

def clean_term(term: str) -> str:
    term = term.strip()
    return term.strip(" '\"").replace('_', ' ').lower()

def load_synonym_dict(csv_path, synonym_column):
    synonym_dict = {}
    # all lower-case
    ignorelist = [
        'go','day','love','most','3','stuff','friend','baby','white',
        'drop','running','pop','phone','water','light','sprite',
        'cheese','juice','chicken','coffee',
        'food','pizza','ghost','boat','truck'
    ]
    ignore_set = set(ignorelist)

    with open(csv_path, newline='', encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            index_term = clean_term(row.get("drug", ""))
            if not index_term:
                continue

            raw = row.get(synonym_column, "")
            raw = raw.strip() 
            if isinstance(raw, str) and raw.strip():
                try:
                    syns = ast.literal_eval(raw)
                except (ValueError, SyntaxError, TypeError):
                    syns = []
            elif isinstance(raw, list):
                syns = raw
            else:
                syns = []

            cleaned = []
            for s in syns:
                cs = clean_term(str(s))
                #########################################
                #########################################
                if len(cs) >= 3 and cs not in ignore_set:
                    cleaned.append(cs)
                #########################################
                #########################################
            cleaned.append(index_term)

            # Dedupe 
            deduped = list(OrderedDict((term, term) for term in cleaned).values())
            synonym_dict[index_term] = deduped
    drugs_with_syns = [drug for drug, terms in synonym_dict.items() if len(terms) > 1]
    print("Drugs with ≥1 synonym ({} total):".format(len(drugs_with_syns)))
    return synonym_dict

def get_regex_and_mapping(synonym_dict):
    term_to_index = {}
    for idx, syns in synonym_dict.items():
        term_to_index[idx] = idx
        for s in syns:
            term_to_index[s] = idx

    # Sort keys by length desc to avoid shorter‐term collisions
    terms = sorted(term_to_index.keys(), key=len, reverse=True)
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b',
        re.IGNORECASE
    )
    print('Total number of synonyms', len(term_to_index))
    return pattern, term_to_index


def match_terms(synonym_column, dataframe, dataframe_column, found_column_name, found_index_column_name):
    """
       ADDS 2 columns
    """
    csv_path = 'data/synonym_lists.csv'
    synonym_dict = load_synonym_dict(csv_path, synonym_column)
    pattern, term_to_index = get_regex_and_mapping(synonym_dict)
    #############################################################################################
    def extract_terms(text):
        if pd.isna(text):
            return [], []

        # findall returns the matched substring; lowercase to key into term_to_index
        matches = pattern.findall(text)
        if not matches:
            return [], []

        found_index = set()
        found_terms = []
        for m in matches:
            key = m.lower()
            idx = term_to_index.get(key)
            if idx:
                found_index.add(idx)
                found_terms.append(key)

        return found_terms, list(found_index)
    #############################################################################################
    #############################################################################################
    dataframe[[found_column_name, found_index_column_name]] = (
        dataframe[dataframe_column]
        .progress_apply(lambda x: pd.Series(extract_terms(x)))
    )
    #############################################################################################
    #############################################################################################
    return dataframe

def create_confusion_matrix(df):
    """
    Creates a confusion matrix based on 'found_index_terms' and 'label' columns.
    
    Parameters:
    - df: DataFrame with 'found_index_terms' and 'label' columns
    
    Returns:
    - confusion_matrix: Dictionary with confusion matrix values
    - cm_array: NumPy array representation of confusion matrix
    """
    # Initialize confusion matrix counts
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    for _, row in df.iterrows():
        # Parse found_index_terms if it's a string representation of a list
        if isinstance(row['found_index_terms'], str):
            try:
                # Safely evaluate the string as a Python expression
                found_terms = literal_eval(row['found_index_terms'])
            except (ValueError, SyntaxError):
                # If evaluation fails, assume it's not a valid list
                found_terms = []
        else:
            # If it's already a list, use it directly
            found_terms = row['found_index_terms']
        
        # Check if found_index_terms has at least one term
        has_term = isinstance(found_terms, list) and len(found_terms) > 0
        
        # Check if label is positive (y or Y)
        if isinstance(row['label'], str):
            is_positive = row['label'].lower() == 't'
        else:
            is_positive = False
        
        # Update confusion matrix
        if has_term and is_positive:
            true_positive += 1
        elif has_term and not is_positive:
            false_positive += 1
        elif not has_term and not is_positive:
            true_negative += 1
        elif not has_term and is_positive:
            false_negative += 1
    
    # Create dictionary for the confusion matrix
    confusion_matrix = {
        'True Positive': true_positive,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'False Negative': false_negative
    }
    
    # Create a 2x2 array for visualization
    cm_array = np.array([
        [true_positive, false_negative],
        [false_positive, true_negative]
    ])
    
    return confusion_matrix, cm_array

def calculate_metrics(confusion_matrix):
    """
    Calculate precision, recall, F1-score, and accuracy from confusion matrix.
    
    Parameters:
    - confusion_matrix: Dictionary with TP, FP, TN, FN values
    
    Returns:
    - metrics: Dictionary with precision, recall, F1-score, and accuracy
    """
    tp = confusion_matrix['True Positive']
    fp = confusion_matrix['False Positive']
    tn = confusion_matrix['True Negative']
    fn = confusion_matrix['False Negative']
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score,
        'Accuracy': accuracy
    }
    
    return metrics

def visualize_confusion_matrix(cm_array):
    """
    Visualize the confusion matrix.
    
    Parameters:
    - cm_array: NumPy array representation of confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def get_confusion_matrix_and_metrics(dataset):
    confusion_matrix, cm_array = create_confusion_matrix(dataset)
    metrics = calculate_metrics(confusion_matrix)

    print("Confusion Matrix:")
    print(f"True Positive: {confusion_matrix['True Positive']}")
    print(f"False Positive: {confusion_matrix['False Positive']}")
    print(f"True Negative: {confusion_matrix['True Negative']}")
    print(f"False Negative: {confusion_matrix['False Negative']}")

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    visualize_confusion_matrix(cm_array)
        
    return 
