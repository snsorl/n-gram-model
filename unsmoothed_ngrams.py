from collections import defaultdict

# Step 1: Read and tokenize the corpus from train.txt
def load_corpus(filename):
    tokens = ["<s>"]  # Add special start token
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            tokens.extend(line.strip().split())  # Split on spaces
    return tokens

# Load the corpus from train.txt
tokens = load_corpus("train.txt")

# Step 2: Count unigrams and bigrams
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)
total_tokens = len(tokens)

for i in range(total_tokens):
    unigram_counts[tokens[i]] += 1
    if i > 0:  # Ensure we don't go out of bounds for bigrams
        bigram_counts[(tokens[i - 1], tokens[i])] += 1

# Step 3: Compute probabilities with Laplace smoothing
V = len(unigram_counts)  # Vocabulary size for smoothing

unigram_probs = {word: (count + 1) / (total_tokens + V) for word, count in unigram_counts.items()}  # Laplace smoothing
bigram_probs = {pair: (count + 1) / (unigram_counts[pair[0]] + V) for pair, count in bigram_counts.items()}  # Smoothed bigram probabilities

# Print results
print("Unigram Probabilities:")
for word, prob in unigram_probs.items():
    print(f"P({word}) = {prob:.6f}")

print("\nBigram Probabilities:")
for pair, prob in bigram_probs.items():
    print(f"P({pair[1]} | {pair[0]}) = {prob:.6f}")
