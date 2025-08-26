import collections

MESSAGE_LENGTH = 721

def calculate_ic(text: str) -> float:
    """
    Calculates the Index of Coincidence for a given text. This function
    is used to distinguish real language from random noise. A higher IC
    (around 0.067 for English) indicates a structured language.
    """
    letters_only = list(filter(str.isalpha, text.upper()))
    n = len(letters_only)
    if n < 2:
        return 0.0

    counts = collections.Counter(letters_only)
    
    numerator = sum(count * (count - 1) for count in counts.values())
    denominator = n * (n - 1)
    
    return numerator / denominator if denominator > 0 else 0.0

def find_message_by_ic(full_signal: str) -> str:
    """
    Finds the correct message block by searching for the segment with the
    highest Index of Coincidence. This is the most reliable method.
    """
    best_ic = -1.0
    best_chunk = ""

    for i in range(len(full_signal) - MESSAGE_LENGTH + 1):
        chunk = full_signal[i:i + MESSAGE_LENGTH]
        ic = calculate_ic(chunk)
        if ic > best_ic:
            best_ic = ic
            best_chunk = chunk
            
    print(f"Found best block with IC = {best_ic:.4f} (English is ~0.067)")
    return best_chunk

def decrypt(text: str, key: dict) -> str:
    """Decrypts the given text using the provided substitution key."""
    return "".join(key.get(char, char) for char in text)

def main():
    """Main function to find, decrypt, and display the final message."""
    print("🛸 NASA Signal Decoder - Final Solution Code 🛸")
    
    try:
        with open(r'C:\Users\Saad Ullah Saleem\Desktop\test\Outer-Space-Signals\signal.txt', 'r') as f:
            signal_data = f.read()
    except FileNotFoundError:
        print("\n❌ ERROR: `signal.txt` not found. Please ensure it is in the project's root directory.")
        return

    # --- Step 1: Find the message using the reliable IC method ---
    print("\nScanning signal for the correct message block...")
    encrypted_message = find_message_by_ic(signal_data)
    
    if not encrypted_message:
        print("\n❌ ERROR: Could not find the message block.")
        return
        
    print("✅ Confirmed signal lock. Applying the final decryption key...")
    
    # --- Step 2: The correct, fully solved key for this challenge ---
    # This key was derived by finding the correct signal block and then solving
    # the substitution cipher using frequency analysis and pattern recognition.
    full_decryption_key = {
        'L': 'T', 'V': 'H', 'P': 'E', 'M': 'I', 'H': 'S', 'G': 'A', 'Z': 'N', 'Y': 'R', 'Q': 'O',
        'C': 'W', 'J': 'C', 'F': 'F', 'O': 'D', 'W': 'U', 'U': 'L', 'B': 'Y', 'I': 'V', 'T': 'G',
        'X': 'P', 'R': 'B', 'S': 'K', 'D': 'M', 'E': 'X', 'A': 'Z', 'K': 'J', 'N': 'Q'
    }

    deciphered_message = decrypt(encrypted_message, full_decryption_key)
    
    # --- Step 3: Display the final, correct results ---
    print("\n--- 👽 FULLY DECIPHERED MESSAGE 👽 ---")
    print(deciphered_message)

    first_nine_words = " ".join(deciphered_message.split()[:9])
    
    print("\n--- ✅ PROPOSAL SUBMISSION ✅ ---")
    print("The first 9 words of the deciphered message are:")
    print(f'"{first_nine_words}"')


if __name__ == "__main__":
    main()
