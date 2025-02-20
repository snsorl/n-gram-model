from collections import defaultdict

# Read the text file and split it into words
def load_corpus(filename):
    tokens = ["<s>"]  # Start with a special start token
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            tokens.extend(line.strip().split())  # Split each line into words
    return tokens

# Load words from train.txt
tokens = load_corpus("train.txt")

# Count how many times each word and word pair appears
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)
total_tokens = len(tokens)

for i in range(total_tokens):
    unigram_counts[tokens[i]] += 1 # Count single words
    if i > 0:  # Ensure we don't go out of bounds for bigrams
        bigram_counts[(tokens[i - 1], tokens[i])] += 1  # Count word pairs

# Calculate probabilities using Laplace smoothing to handle unseen words
V = len(unigram_counts)  # Vocabulary size for smoothing

# Compute probabilities with smoothing to avoid zero probabilities
unigram_probs = {word: (count + 1) / (total_tokens + V) for word, count in unigram_counts.items()} 
bigram_probs = {pair: (count + 1) / (unigram_counts[pair[0]] + V) for pair, count in bigram_counts.items()} 

# Print results
print("Unigram Probabilities:")
for word, prob in unigram_probs.items():
    print(f"P({word}) = {prob:.6f}")

print("\nBigram Probabilities:")
for pair, prob in bigram_probs.items():
    print(f"P({pair[1]} | {pair[0]}) = {prob:.6f}")
