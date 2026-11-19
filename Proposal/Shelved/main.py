import math
import random
from collections import Counter

class StatisticalSandhiSplitter:
    def __init__(self, p_phi=0.5):
        self.p_phi = p_phi  # Probability of a boundary rule occurring
        self.phoneme_probs = {}
        self.total_phonemes = 0

    def train(self, unsandhied_words):
        """Train unigram phoneme probabilities from clean, unsandhied text."""
        counter = Counter()
        for word in unsandhied_words:
            for char in word:
                counter[char] += 1
        
        self.total_phonemes = sum(counter.values())
        # Add smoothing and convert counts to probabilities P(a)
        vocab_size = len(counter)
        alpha = 0.1
        for char, count in counter.items():
            self.phoneme_probs[char] = (count + alpha) / (self.total_phonemes + alpha * vocab_size)
        
        # Default fallback for unseen phonemes
        self.default_prob = alpha / (self.total_phonemes + alpha * vocab_size)

    def morph_log_probability(self, morph):
        """
        Computes the log probability (surprisal-based score) of a morph 
        based on the unigram phoneme generator model:
        Ps(c) = p_phi * (1 - p_phi)^(d-1) * product(P(a_j))
        """
        if not morph:
            return float('-inf')
        
        d = len(morph)
        # Log probability of the boundary rules
        log_prob = math.log(self.p_phi) + (d - 1) * math.log(1.0 - self.p_phi)
        
        # Sum of log probabilities of individual phonemes
        for char in morph:
            p_a = self.phoneme_probs.get(char, self.default_prob)
            log_prob += math.log(p_a)
            
        return log_prob

    def score_split(self, morphs):
        """Total score for a list of morphs (higher is better, minimizing global surprisal)."""
        return sum(self.morph_log_probability(m) for m in morphs)

    def get_all_splits(self, text):
        """Generate all possible binary split positions for a chunk of text."""
        # For a string of length L, there are 2^(L-1) possible split configurations.
        # Here we yield a few candidate segmentations for demonstration.
        n = len(text)
        if n <= 1:
            return [[text]]
        
        splits = []
        # Let's generate a set of random valid cuts or systematic cuts for iterative testing
        for i in range(1, n):
            splits.append([text[:i], text[i:]])
        splits.append([text]) # Unsplit baseline
        return splits

    def iterative_segment(self, sandhied_chunk, iterations=50):
        """
        An iterative stochastic search (similar to MCMC/Gibbs sampling steps) 
        that tweaks boundary positions to maximize overall sequence probability 
        (minimizing surprisal).
        """
        # Initialize with a random split position
        n = len(sandhied_chunk)
        if n <= 1:
            return [sandhied_chunk]
            
        current_split_idx = random.randint(1, n - 1)
        current_morphs = [sandhied_chunk[:current_split_idx], sandhied_chunk[current_split_idx:]]
        current_score = self.score_split(current_morphs)

        # Iterative stochastic descent / ascent
        for _ in range(iterations):
            # Propose a neighbor state by shifting the split point slightly or adding/removing a cut
            possible_indices = list(range(1, n))
            if not possible_indices:
                break
            new_idx = random.choice(possible_indices)
            proposed_morphs = [sandhied_chunk[:new_idx], sandhied_chunk[new_idx:]]
            proposed_score = self.score_split(proposed_morphs)

            # Accept if better, or use a Metropolis-style acceptance criterion (stochastic relative)
            # Since we want to maximize probability, higher score is favored
            score_diff = proposed_score - current_score
            if score_diff > 0 or random.random() < math.exp(score_diff / 2.0): # Temperature scaling
                current_morphs = proposed_morphs
                current_score = proposed_score

        return current_morphs

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Mock training data (unsandhied clean morphs)
    clean_lexicon = ["deva", "alaya", "rama", "ishana", "narayan", "iti", "api"]
    
    splitter = StatisticalSandhiSplitter(p_phi=0.4)
    splitter.train(clean_lexicon)
    
    # 2. Test chunk containing a blended sandhi boundary (e.g., deva + alaya -> devalaya)
    test_chunk = "devAlaya"
    
    # 3. Run iterative statistical optimization
    best_split = splitter.iterative_segment(test_chunk, iterations=100)
    print(f"Original Sandhied Text: {test_chunk}")
    print(f"Statistically Split Morphists: {best_split}")