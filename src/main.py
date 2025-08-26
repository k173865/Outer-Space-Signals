import sys
import math
import random
import string
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ==============================
# Config / constants
# ==============================
TARGET_LEN = 721
TARGET_FREQ_ORDER = list("EATOIRSNHU")  # given by the challenge
ALPHABET = list(string.ascii_uppercase)
SPACE = " "

# Frequent English short words and function words to boost scoring
COMMON_WORDS = [
    "THE", "AND", "OF", "TO", "IN", "IS", "IT", "YOU", "THAT", "FOR", "ON",
    "WITH", "AS", "ARE", "THIS", "BE", "OR", "BY", "AN", "FROM", "AT", "NOT",
    "HAVE", "HE", "SHE", "THEY", "WE", "I", "BUT", "ALL", "ONE", "ABOUT", "IF",
    "CAN", "THERE", "WHICH", "WOULD", "THEIR", "HAS", "MORE", "WILL", "WHO", "WHAT",
]

# A compact set of high-frequency English bigrams/trigrams (used as soft signal).
COMMON_NGRAMS = [
    # bigrams
    "TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT", "EN", "ND", "TI", "ES", "OR", "TE", "OF", "ED", "IS", "IT", "AL", "AR", "ST", "TO", "NT", "NG",
    # trigrams
    "THE", "ING", "AND", "HER", "ERE", "ENT", "THA", "NTH", "WAS", "ETH", "FOR", "DTH", "HES", "HIS", "NOT", "YOU", "ITH", "ALL", "ARE",
]

# ==============================
# Utility functions
# ==============================

def clean_text(s: str) -> str:
    """Keep only A-Z and space, as per problem statement (input should already match)."""
    return ''.join(ch for ch in s if ch == SPACE or ('A' <= ch <= 'Z'))


def index_of_coincidence(block: str) -> float:
    """Compute Index of Coincidence for a block (spaces ignored)."""
    letters = [c for c in block if c != SPACE]
    n = len(letters)
    if n < 2:
        return 0.0
    counts = Counter(letters)
    num = sum(v * (v - 1) for v in counts.values())
    den = n * (n - 1)
    return num / den if den else 0.0


def freq_order(s: str) -> List[str]:
    """Return letters sorted by descending frequency (ties by alpha)."""
    cnt = Counter(c for c in s if c != SPACE)
    return [k for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]


def initial_key_from_freq(cipher: str) -> Dict[str, str]:
    """Seed a monoalphabetic key by mapping most frequent cipher letters to TARGET_FREQ_ORDER."""
    order = freq_order(cipher)
    key = {}
    # Map top to target
    for i, c in enumerate(order):
        if i < len(TARGET_FREQ_ORDER):
            key[c] = TARGET_FREQ_ORDER[i]
    # Map the remaining letters greedily by remaining alphabet
    used = set(key.values())
    remaining_plain = [ch for ch in ALPHABET if ch not in used]
    for c in ALPHABET:
        if c not in key:
            key[c] = remaining_plain.pop(0)
    return key


def apply_key(text: str, key: Dict[str, str]) -> str:
    table = {**{SPACE: SPACE}, **key}
    return ''.join(table.get(ch, ch) for ch in text)


# ==============================
# Scoring
# ==============================

def ngram_score(text: str) -> float:
    """Soft score based on occurrences of frequent n-grams (case: already uppercase)."""
    t = text.replace(SPACE, "")
    score = 0.0
    for ng in COMMON_NGRAMS:
        # add log(1+count) to avoid dominating by length alone
        c = t.count(ng)
        if c:
            score += math.log(1 + c)
    return score


def word_score(text: str) -> float:
    """Soft score for presence of common function words."""
    words = text.split()
    bag = Counter(words)
    return sum(bag.get(w, 0) for w in COMMON_WORDS)


def freq_order_bonus(text: str) -> float:
    letters_only = [c for c in text if c != SPACE]
    if not letters_only:
        return 0.0
    order = [k for k, _ in sorted(Counter(letters_only).items(), key=lambda kv: (-kv[1], kv[0]))]
    bonus = 0.0
    for i, expected in enumerate(TARGET_FREQ_ORDER):
        if i < len(order) and order[i] == expected:
            bonus += 1.0
    return bonus


def english_likeness(text: str) -> float:
    # Combine components. We intentionally weigh words higher; adjust as needed.
    return 2.0 * word_score(text) + 1.2 * ngram_score(text) + 0.5 * freq_order_bonus(text)


# ==============================
# Hill-climbing / simulated annealing for substitution
# ==============================

def random_key() -> Dict[str, str]:
    shuffled = ALPHABET.copy()
    random.shuffle(shuffled)
    return dict(zip(ALPHABET, shuffled))


def key_swap(key: Dict[str, str], a: str, b: str) -> Dict[str, str]:
    new_key = key.copy()
    # swap images of a and b (cipher letters map to plain letters)
    pa, pb = new_key[a], new_key[b]
    new_key[a], new_key[b] = pb, pa
    return new_key


def refine_with_hill_climb(cipher: str, seed_key: Dict[str, str], iters: int = 8000, temp_start: float = 2.0, temp_end: float = 0.05) -> Tuple[str, Dict[str, str], float]:
    """Simulated annealing: swap cipher->plain assignments to maximize english_likeness."""
    current_key = seed_key.copy()
    current_plain = apply_key(cipher, current_key)
    current_score = english_likeness(current_plain)

    best_key = current_key
    best_plain = current_plain
    best_score = current_score

    for t in range(1, iters + 1):
        # cooling schedule
        frac = t / iters
        temp = temp_start * (1 - frac) + temp_end * frac
        a, b = random.sample(ALPHABET, 2)
        trial_key = key_swap(current_key, a, b)
        trial_plain = apply_key(cipher, trial_key)
        trial_score = english_likeness(trial_plain)
        delta = trial_score - current_score
        if delta >= 0 or math.exp(delta / max(1e-9, temp)) > random.random():
            current_key, current_plain, current_score = trial_key, trial_plain, trial_score
            if trial_score > best_score:
                best_key, best_plain, best_score = trial_key, trial_plain, trial_score
    return best_plain, best_key, best_score


def crack_substitution(cipher: str, restarts: int = 8, iters: int = 10000) -> Tuple[str, Dict[str, str], float]:
    """Multiple-restart annealing initialized from frequency mapping and random keys."""
    best_plain, best_key, best_score = None, None, -1e9

    # 1) Start from frequency-based seed biased by given TARGET_FREQ_ORDER
    seed = initial_key_from_freq(cipher)
    p, k, s = refine_with_hill_climb(cipher, seed, iters=iters)
    if s > best_score:
        best_plain, best_key, best_score = p, k, s

    # 2) Additional random restarts for robustness
    for _ in range(restarts):
        seed = random_key()
        p, k, s = refine_with_hill_climb(cipher, seed, iters=iters // 2)
        if s > best_score:
            best_plain, best_key, best_score = p, k, s

    return best_plain, best_key, best_score


# ==============================
# Candidate window search
# ==============================

def candidate_indices_by_ic(signal: str, length: int, stride: int = 23, top_k: int = 20) -> List[int]:
    """Return candidate start indices whose IC is closest to English (~0.066)."""
    target_ic = 0.066
    scored: List[Tuple[float, int]] = []
    for i in range(0, max(0, len(signal) - length + 1), stride):
        block = signal[i: i + length]
        ic = index_of_coincidence(block)
        scored.append((abs(ic - target_ic), i))
    scored.sort(key=lambda x: x[0])
    return [i for _, i in scored[:top_k]]


def refine_indices_around(starts: List[int], length: int, radius: int = 60) -> List[int]:
    """Expand candidate starts by exploring a small neighborhood around each index."""
    refined = set()
    for s in starts:
        for j in range(max(0, s - radius), s + radius + 1):
            refined.add(j)
    return sorted(refined)


# ==============================
# Main pipeline
# ==============================

def main():
    """Main function to run the program."""
    print("🛸 NASA Signal Decoder - Deciphering Messages from Planet Dyslexia 🛸")

    # Load signal
    path = r"C:\Users\Saad Ullah Saleem\Desktop\test\Outer-Space-Signals\signal.txt"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    try:
        with open(path, "r", encoding="utf-8") as f:
            signal_text = clean_text(f.read().strip().upper())
    except FileNotFoundError:
        print(f"⚠️ {path} not found. Please provide the 64KB file (uppercase letters and spaces).")
        return

    if len(signal_text) < TARGET_LEN:
        print("❌ Signal file is shorter than the expected message length.")
        return

    print("Analyzing 64KB of alien signals…")

    # 1) Fast coarse scan by Index of Coincidence
    print("🔭 Scanning for likely 721-char windows (IC filter)…")
    coarse_candidates = candidate_indices_by_ic(signal_text, TARGET_LEN, stride=23, top_k=24)

    # 2) Refine in a neighborhood to avoid stride misses
    refined_indices = refine_indices_around(coarse_candidates, TARGET_LEN, radius=50)

    # 3) Score refined windows by simple English-likeness from freq-mapped seed
    print("🧪 Ranking refined candidates by quick English-likeness…")
    window_scores: List[Tuple[float, int]] = []
    for i in refined_indices:
        block = signal_text[i:i+TARGET_LEN]
        seed_key = initial_key_from_freq(block)
        seed_plain = apply_key(block, seed_key)
        score = english_likeness(seed_plain)
        window_scores.append((score, i))
    window_scores.sort(reverse=True)

    # Take top N for full cracking
    top_windows = window_scores[:8]

    print("🚀 Cracking top candidate windows with simulated annealing… (this may take a moment)")
    best_overall = (-1e9, "", {}, -1)  # (score, plaintext, key, start_index)

    for rank, (pre_score, idx) in enumerate(top_windows, 1):
        block = signal_text[idx:idx+TARGET_LEN]
        plain, key, score = crack_substitution(block, restarts=6, iters=12000)
        total_score = score
        if total_score > best_overall[0]:
            best_overall = (total_score, plain, key, idx)
        print(f"   • Candidate {rank}: start={idx}, pre_score={pre_score:.2f}, cracked_score={total_score:.2f}")

    if best_overall[0] <= -1e8:
        print("❌ No valid message detected.")
        return

    best_plain, best_key, best_start = best_overall[1], best_overall[2], best_overall[3]

    # Output results
    first9 = ' '.join(best_plain.split()[:9])
    print("\n==================== RESULT ====================")
    print(first9)
    print("================================================")

    # Persist outputs
    with open("decoded_message.txt", "w", encoding="utf-8") as f:
        f.write(best_plain)
    with open("mapping.txt", "w", encoding="utf-8") as f:
        # Save cipher->plain mapping
        for c in ALPHABET:
            f.write(f"{c}->{best_key[c]}\n")
    with open("window_info.txt", "w", encoding="utf-8") as f:
        f.write(f"start_index={best_start}\nlength={TARGET_LEN}\n")

    print("Saved: decoded_message.txt, mapping.txt, window_info.txt")


if __name__ == "__main__":
    random.seed(1337)
    main()
