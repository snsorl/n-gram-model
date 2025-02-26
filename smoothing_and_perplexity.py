import os
import math
from collections import defaultdict


# Step 1: Load & Preprocess Data

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f" Loaded {len(lines)} reviews from {file_path}")
    return [line.strip().split() for line in lines]

def build_vocab(train_data, threshold=1):
    word_counts = defaultdict(int)

    for sentence in train_data:
        for word in sentence:
            word_counts[word] += 1

    # Keep words that appear more than threshold, replace others with <UNK>
    vocab = {word for word in word_counts if word_counts[word] > threshold}
    vocab.add("<UNK>")

    def replace_unknown(sentence):
        return [word if word in vocab else "<UNK>" for word in sentence]

    processed_data = [replace_unknown(sentence) for sentence in train_data]

    return processed_data, vocab

# Step 2: Train Unigram & Bigram Models

def train_ngram_models(train_data):
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    total_words = 0

    for sentence in train_data:
        for i in range(len(sentence)):
            unigram_counts[sentence[i]] += 1
            total_words += 1
            if i > 0:
                bigram_counts[(sentence[i - 1], sentence[i])] += 1

    # Compute probabilities
    unigram_probs = {word: count / total_words for word, count in unigram_counts.items()}
    bigram_probs = {(w1, w2): count / unigram_counts[w1] for (w1, w2), count in bigram_counts.items()}

    return unigram_probs, bigram_probs, unigram_counts, bigram_counts

# Step 3: Implement Smoothing

def laplace_smoothing(bigram_counts, unigram_counts, vocab_size):
    smoothed_probs = {}

    for (w1, w2), count in bigram_counts.items():
        smoothed_probs[(w1, w2)] = (count + 1) / (unigram_counts[w1] + vocab_size)

    return smoothed_probs

def add_k_smoothing(bigram_counts, unigram_counts, vocab_size, k=0.1):
    smoothed_probs = {}

    for (w1, w2), count in bigram_counts.items():
        smoothed_probs[(w1, w2)] = (count + k) / (unigram_counts[w1] + k * vocab_size)

    return smoothed_probs


# Step 4: Compute Perplexity

def compute_perplexity(test_data, smoothed_probs, unigram_probs, vocab_size):
    total_log_prob = 0
    total_words = 0

    for sentence in test_data:
        for i in range(1, len(sentence)):
            w1, w2 = sentence[i - 1], sentence[i]
            prob = smoothed_probs.get((w1, w2), 1 / (unigram_probs.get(w1, 0) * vocab_size))
            total_log_prob += -math.log(prob)
            total_words += 1

    return math.exp(total_log_prob / total_words)

# Compute Unigram Perplexity
def compute_unigram_perplexity(test_data, unigram_probs):
    total_log_prob = 0
    total_words = 0

    for sentence in test_data:
        for word in sentence:
            prob = unigram_probs.get(word, 1e-6)  # Small probability for unknown words
            total_log_prob += -math.log(prob)
            total_words += 1

    return math.exp(total_log_prob / total_words)

# Step 5: Main Execution

# Set dataset path
dataset_dir = os.path.join(os.getcwd())

# Train Models
train_path = os.path.join(dataset_dir, "train.txt")
print(f"Looking for train file at: {train_path}")
train_data = load_data(train_path)
processed_train, vocab = build_vocab(train_data)
unigram_probs, bigram_probs, unigram_counts, bigram_counts = train_ngram_models(processed_train)

#Validation 
valid_path = os.path.join(dataset_dir, "val.txt")
print(f"Looking for validation file at: {valid_path}")
valid_data = load_data(valid_path)
processed_valid = [[word if word in vocab else "<UNK>" for word in sentence] for sentence in valid_data]

# Apply Smoothing
vocab_size = len(vocab)
laplace_probs = laplace_smoothing(bigram_counts, unigram_counts, vocab_size)

# Define a range of k values to test
k_values = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

# Compute Perplexity on Validation Set
perplexity_unigram_valid = compute_unigram_perplexity(processed_valid, unigram_probs)
perplexity_unsmoothed_valid = compute_perplexity(processed_valid, bigram_probs, unigram_probs, vocab_size)
perplexity_laplace_valid = compute_perplexity(processed_valid, laplace_probs, unigram_probs, vocab_size)

# Print Final Results
print("\n      RESULTS")
print(f"Validation Perplexity (Unigram): {perplexity_unigram_valid:.6f}")
print(f"Validation Perplexity (Bigram, Unsmoothed): {perplexity_unsmoothed_valid:.6f}")
print(f"Validation Perplexity (Bigram, Laplace Smoothing): {perplexity_laplace_valid:.6f}")

# Test each k value
for k in k_values:
    add_k_probs = add_k_smoothing(bigram_counts, unigram_counts, vocab_size, k=k)
    perplexity = compute_perplexity(processed_valid, add_k_probs, unigram_probs, vocab_size)
    print(f"Validation Perplexity (Add-k Smoothing, k={k}): {perplexity:.6f}")
