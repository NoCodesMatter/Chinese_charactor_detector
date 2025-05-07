import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

# Dataset class for Chinese spelling correction
class ChineseTextDataset(Dataset):
    def __init__(self, wrong_texts, right_texts, vocab, max_len=128):
        """
        Initializes the dataset with the wrong and correct text pairs, vocabulary, and maximum sequence length.
        """
        self.wrong_texts = wrong_texts
        self.right_texts = right_texts
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.wrong_texts)
    
    def __getitem__(self, idx):
        """
        Returns a single sample (wrong and correct text as tensors).
        """
        wrong_text = self.wrong_texts[idx]
        right_text = self.right_texts[idx]
        
        # Convert text into a sequence of indices using the vocabulary
        wrong_indices = [self.vocab.get(char, self.vocab['<UNK>']) for char in wrong_text]
        right_indices = [self.vocab.get(char, self.vocab['<UNK>']) for char in right_text]
        
        return {
            'wrong_text': torch.tensor(wrong_indices, dtype=torch.long),
            'right_text': torch.tensor(right_indices, dtype=torch.long),
            'length': len(wrong_indices)
        }

# Model class for the spelling corrector
class SpellingCorrector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        """
        Initializes the spelling correction model with embedding, BiLSTM, and output layers.
        """
        super(SpellingCorrector, self).__init__()
        # Embedding layer: maps input indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # BiLSTM: Processes embedded sequences to capture contextual information
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer: Projects LSTM outputs to vocabulary size
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, lengths):
        """
        Forward pass of the model.
        """
        # Embed input indices to dense vectors
        embedded = self.embedding(x)
        # Convert lengths tensor to CPU for packing
        lengths = lengths.cpu()
        # Pack the embedded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        # Process packed sequence with BiLSTM
        lstm_out, _ = self.lstm(packed)
        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Project LSTM outputs to vocabulary size
        output = self.fc(lstm_out)
        
        return output

# Function to build a vocabulary from the dataset
def build_vocab(texts):
    """
    Builds a vocabulary dictionary from the given texts.
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens for padding and unknown characters
    idx = 2
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = idx
                idx += 1
    return vocab

# Function to load data from a JSONL file
def load_data(jsonl_file):
    """
    Loads the dataset from a JSONL file containing wrong and correct text pairs.
    """
    wrong_texts = []
    right_texts = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            wrong_texts.append(data['wrong'])
            right_texts.append(data['correct'])
    
    return wrong_texts, right_texts

# Custom collate function for batching
def collate_fn(batch):
    """
    Pads sequences to the same length for batching.
    """
    max_len = max(item['length'] for item in batch)
    for item in batch:
        item['wrong_text'] = torch.cat([item['wrong_text'], torch.tensor([0] * (max_len - len(item['wrong_text'])), dtype=torch.long)])
        item['right_text'] = torch.cat([item['right_text'], torch.tensor([0] * (max_len - len(item['right_text'])), dtype=torch.long)])
    wrong_texts = torch.stack([item['wrong_text'] for item in batch])
    right_texts = torch.stack([item['right_text'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    return {'wrong_text': wrong_texts, 'right_text': right_texts, 'length': lengths}

# Function to train the model
def train_model(model, train_loader, device, num_epochs=10):
    """
    Trains the model using the given training data loader.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens during loss calculation
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            wrong_texts = batch['wrong_text'].to(device, dtype=torch.long)
            right_texts = batch['right_text'].to(device, dtype=torch.long)
            lengths = batch['length'].to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            # Get model outputs
            outputs = model(wrong_texts, lengths)
            
            # Reshape tensors to match dimensions for loss calculation
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)
            targets = right_texts.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print batch loss
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return model

# Function to correct a given text using the trained model
def correct_text(model, text, vocab, device, max_len=128):
    """
    Corrects a given text using the trained model.
    """
    # Create reverse vocabulary
    idx2char = {idx: char for char, idx in vocab.items()}
    
    # Prepare input
    indices = [vocab.get(char, vocab['<UNK>']) for char in text]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    lengths = torch.tensor([min(len(text), max_len)], dtype=torch.long)
    
    # Get model output
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor, lengths)
        predictions = outputs.argmax(dim=-1)[0]
    
    # Convert predictions to text
    corrected_text = ''.join([idx2char[idx.item()] for idx in predictions[:len(text)]])
    return corrected_text

# Interactive correction mode for user input
def interactive_correction(model, vocab, device):
    """
    Allows interactive correction of user-provided texts.
    """
    print("\n=== Welcome to the Chinese Spelling Correction System ===")
    print("Type 'q' or 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter text to check: ").strip()
        
        if user_input.lower() in ['q', 'quit']:
            print("Thank you for using the system. Goodbye!")
            break
        
        if not user_input:
            print("Input is empty. Please try again.")
            continue
        
        try:
            corrected = correct_text(model, user_input, vocab, device)
            print("\nResult:")
            print(f"Original: {user_input}")
            print(f"Corrected: {corrected}")
            
            if user_input != corrected:
                print("\nSuggestions:")
                for i, (orig_char, corr_char) in enumerate(zip(user_input, corrected)):
                    if orig_char != corr_char:
                        print(f"Position {i+1}: '{orig_char}' -> '{corr_char}'")
            else:
                print("\nNo errors found!")
                
        except Exception as e:
            print(f"Error during processing: {str(e)}")

# Main function to run the program
def main():
    """
    Main function to load data, train the model, and start interactive correction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        checkpoint = torch.load('spelling_corrector.pth', map_location=device)
        vocab = checkpoint['vocab']
        model = SpellingCorrector(len(vocab)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded the trained model!")
        interactive_correction(model, vocab, device)
    except FileNotFoundError:
        print("No trained model found. Starting training...")
        
        wrong_texts, right_texts = load_data('data/output.jsonl')
        print(f"Loaded {len(wrong_texts)} text pairs")
        
        vocab = build_vocab(wrong_texts + right_texts)
        print(f"Vocabulary size: {len(vocab)}")
        
        dataset = ChineseTextDataset(wrong_texts, right_texts, vocab)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        model = SpellingCorrector(len(vocab)).to(device)
        model = train_model(model, train_loader, device)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab
        }, 'spelling_corrector.pth')
        
        print("\nTraining complete! Entering interactive mode:")
        interactive_correction(model, vocab, device)

if __name__ == "__main__":
    main()
