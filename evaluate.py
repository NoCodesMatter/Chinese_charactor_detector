import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from spelling_corrector import SpellingCorrector


def load_test_data(filename):
    test_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data.append((data['wrong'], data['correct']))
    return test_data


def evaluate_model(corrector, test_data):
    mse_scores = []
    accuracy_scores = []
    error_correction_scores = []

    for wrong_text, right_text in test_data:
        corrected_text, _ = corrector.correct(wrong_text)

        # Calculate MSE and accuracy
        mse = calculate_mse(right_text, corrected_text, wrong_text)
        accuracy, error_correction = calculate_accuracy(right_text, corrected_text, wrong_text)

        mse_scores.append(mse)
        accuracy_scores.append(accuracy)
        error_correction_scores.append(error_correction)

    # Output average values
    avg_mse = np.mean(mse_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_error_correction = np.mean(error_correction_scores)

    print(f'Average MSE: {avg_mse}')
    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average Error Correction: {avg_error_correction}')

    return avg_mse, avg_accuracy, avg_error_correction


def evaluate_accuracy(inp, out, corrected):
    correct_mods = 0
    total_cg = 0
    for i, o, c in zip(inp, out, corrected):
        if i != c:
            total_cg += 1
            if o == c:
                correct_mods += 1
    total_error = 0
    correct_cg = 0
    for o, n, c in zip(inp, out, corrected):
        if i != o:
            total_error += 1
            if o == c:
                correct_cg += 1
    return (correct_mods / total_cg if total_cg > 0 else 0,
            correct_cg / total_error if total_error > 0 else 0)


def calculate_pinyin_similarity(char1, char2):
    # Get the pinyin of the characters
    pinyin1 = ''.join([item[0] for item in pinyin(char1, style=Style.NORMAL)])
    pinyin2 = ''.join([item[0] for item in pinyin(char2, style=Style.NORMAL)])

    # Calculate the similarity between the pinyin strings
    similarity = SequenceMatcher(None, pinyin1, pinyin2).ratio()
    return similarity


def calculate_shape_similarity(char1, char2):
    # Get the Unicode encoding of the characters
    codepoint1 = ord(char1)
    codepoint2 = ord(char2)

    # Shape similarity: based on Unicode distance
    similarity = max(0, 1 - abs(codepoint1 - codepoint2) / 10000)  # Normalize distance to [0, 1]
    return similarity


def calculate_character_difference(char1, char2):
    pinyin_similarity = calculate_pinyin_similarity(char1, char2)
    font_similarity = calculate_shape_similarity(char1, char2)
    difference = 2 - pinyin_similarity - font_similarity
    return difference


def calculate_mse(correct_text, predicted_text, noisy_text):
    total_diff = 0
    total_mods = 0
    for char1, char2, char3 in zip(correct_text, predicted_text, noisy_text):
        if char3 != char2:  # If the original text and the predicted text differ, it indicates a modification
            total_mods += 1
            total_diff += calculate_character_difference(char1, char2) ** 2

    mse = total_diff / total_mods if total_mods > 0 else 0
    return mse


# Calculate model correction accuracy
# (correct modifications/total modifications and correctly modified erroneous characters/total erroneous characters)
def calculate_accuracy(correct_text, predicted_text, noisy_text):
    correct_mods = 0
    total_mods = 0
    correct_errors = 0
    total_errors = 0
    for c1, c2, c3 in zip(correct_text, predicted_text, noisy_text):
        if c3 != c2:  # If the original text and the predicted text differ, it indicates a modification
            total_mods += 1
            if c1 == c2:  # If the corrected text matches the correct text, it indicates a correct modification
                correct_mods += 1
        if c1 != c3:  # If the original text differs from the correct text, it indicates an error in the original text
            total_errors += 1
            if c1 == c2:  # If the corrected text matches the correct text, it indicates a correct error correction
                correct_errors += 1
    return (correct_mods / total_mods if total_mods > 0 else 0,
            correct_errors / total_errors if total_errors > 0 else 0)


# Evaluate the model
def evaluate_model1(jsonl_file_path, model_path, device):
    mse_scores = []
    accuracy_scores = []
    error_correction_scores = []

    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    model = SpellingCorrector(len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create reverse vocabulary dictionary
    idx2char = {idx: char for char, idx in vocab.items()}

    # Read the jsonl file line by line
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            correct_text = data['correct']  # Correct text
            noisy_text = data['wrong']  # Noisy text

            # Prepare input
            indices = [vocab.get(char, vocab['<UNK>']) for char in noisy_text]
            if len(indices) < 128:
                indices += [vocab['<PAD>']] * (128 - len(indices))
            else:
                indices = indices[:128]

            input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
            lengths = torch.tensor([min(len(noisy_text), 128)], dtype=torch.long)

            # Use the model to correct the noisy text
            with torch.no_grad():
                outputs = model(input_tensor, lengths)
                predictions = outputs.argmax(dim=-1)[0]

            # Convert back to text
            predicted_text = ''.join([idx2char[idx.item()] for idx in predictions[:len(noisy_text)]])

            # Calculate MSE and accuracy
            mse = calculate_mse(correct_text, predicted_text, noisy_text)
            accuracy, error_correction = calculate_accuracy(correct_text, predicted_text, noisy_text)

            mse_scores.append(mse)
            accuracy_scores.append(accuracy)
            error_correction_scores.append(error_correction)

    # Output average values
    avg_mse = np.mean(mse_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_error_correction = np.mean(error_correction_scores)

    return avg_mse, avg_accuracy, avg_error_correction


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_data_path = 'output_test.jsonl'
    model_path = 'spelling_corrector.pth'

    # Evaluate the model
    avg_mse, avg_accuracy, avg_error_correction = evaluate_model1(test_data_path, model_path, device)
    print(f'Average MSE: {avg_mse}')
    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average Error Correction: {avg_error_correction}')
