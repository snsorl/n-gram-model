import os
import math
from collections import defaultdict

# ===========================
# Step 1: Load & Preprocess Data
# ===========================

def load_data(file_path):
    """ Reads the dataset and tokenizes text """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f" Loaded {len(lines)} reviews from {file_path}")
    return [line.strip().split() for line in lines]

def build_vocab(train_data, threshold=1):
    """ Create a vocabulary and replace low-frequency words with <UNK> """
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

# ===========================
# Step 2: Train Unigram & Bigram Models
# ===========================

def train_ngram_models(train_data):
    """ Train Unigram and Bigram models """
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

# ===========================
# Step 3: Implement Smoothing
# ===========================

def laplace_smoothing(bigram_counts, unigram_counts, vocab_size):
    """ Apply Laplace (Add-1) smoothing to bigram probabilities """
    smoothed_probs = {}

    for (w1, w2), count in bigram_counts.items():
        smoothed_probs[(w1, w2)] = (count + 1) / (unigram_counts[w1] + vocab_size)

    return smoothed_probs

def add_k_smoothing(bigram_counts, unigram_counts, vocab_size, k=0.1):
    """ Apply Add-k smoothing to bigram probabilities """
    smoothed_probs = {}

    for (w1, w2), count in bigram_counts.items():
        smoothed_probs[(w1, w2)] = (count + k) / (unigram_counts[w1] + k * vocab_size)

    return smoothed_probs

# ===========================
# Step 4: Compute Perplexity
# ===========================

def compute_perplexity(test_data, smoothed_probs, unigram_probs, vocab_size):
    """ Calculate Perplexity for a given model """
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
    """ Compute Perplexity for a unigram model """
    total_log_prob = 0
    total_words = 0

    for sentence in test_data:
        for word in sentence:
            prob = unigram_probs.get(word, 1e-6)  # Small probability for unknown words
            total_log_prob += -math.log(prob)
            total_words += 1

    return math.exp(total_log_prob / total_words)

# ===========================
# Step 5: Main Execution
# ===========================

# Set dataset paths (Update these if needed)
dataset_dir = os.path.join(os.getcwd())
train_path = os.path.join(dataset_dir, "train.txt")
valid_path = os.path.join(dataset_dir, "val.txt")

# Verify dataset paths
print(f"Looking for train file at: {train_path}")
print(f"Looking for validation file at: {valid_path}")

# Load data
train_data = load_data(train_path)
valid_data = load_data(valid_path)

# Preprocessing: Handle Unknown Words
processed_train, vocab = build_vocab(train_data)
processed_valid = [[word if word in vocab else "<UNK>" for word in sentence] for sentence in valid_data]

# Train Models
unigram_probs, bigram_probs, unigram_counts, bigram_counts = train_ngram_models(processed_train)

# Apply Smoothing
vocab_size = len(vocab)
laplace_probs = laplace_smoothing(bigram_counts, unigram_counts, vocab_size)
add_k_probs = add_k_smoothing(bigram_counts, unigram_counts, vocab_size, k=0.01)

# Compute Perplexity on Training and Validation Sets
perplexity_unigram_valid = compute_unigram_perplexity(processed_valid, unigram_probs)
perplexity_unsmoothed_valid = compute_perplexity(processed_valid, bigram_probs, unigram_probs, vocab_size)
perplexity_laplace_valid = compute_perplexity(processed_valid, laplace_probs, unigram_probs, vocab_size)
perplexity_add_k_valid = compute_perplexity(processed_valid, add_k_probs, unigram_probs, vocab_size)

# Print Final Results
print("\n------------ RESULTS --------------")
print(f"Validation Perplexity (Unigram): {perplexity_unigram_valid:.6f}")
print(f"Validation Perplexity (Bigram, Unsmoothed): {perplexity_unsmoothed_valid:.6f}")
print(f"Validation Perplexity (Bigram, Laplace Smoothing): {perplexity_laplace_valid:.6f}")
print(f"Validation Perplexity (Bigram, Add-k Smoothing, k=0.01): {perplexity_add_k_valid:.6f}")
print("------------------------------")
