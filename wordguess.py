import datetime
import difflib
import json
import os
import random
import sys
import numpy as np
import spacy

# 1. Load spaCy English Medium Model
print("Loading Natural Language Processing (NLP) model. Please wait...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("\n❌ Error: The 'en_core_web_md' model is not installed.")
    print("Please run 'python -m spacy download en_core_web_md' in your terminal and try again.")
    sys.exit(1)

# 2. Curated Target Word Pool
TARGET_WORD_POOL = [
    "apple", "banana", "airplane", "bicycle", "apartment", "architect", "universe", "galaxy",
    "butter", "cheese", "chocolate", "restaurant", "kitchen", "refrigerator", "computer", "smartphone",
    "library", "university", "education", "student", "teacher", "classroom", "hospital", "doctor",
    "nurse", "medicine", "disease", "avalanche", "volcano", "earthquake", "weather",
    "climate", "environment", "pollution", "government", "president", "election", "democracy", "freedom",
    "justice", "lawyer", "court", "soldier", "weapon", "victory", "tragedy", "comedy",
    "concert", "stadium", "audience", "musician", "painting", "sculpture", "gallery", "photography",
    "industry", "factory", "economy", "investment", "currency", "bankruptcy", "advertisement", "marketing",
    "adventure", "journey", "passport", "luggage", "destination", "tourism", "wilderness", "expedition",
    "shadow", "reflection", "illusion", "miracle", "mystery", "secret", "gossip", "reputation",
    "friendship", "marriage", "childhood", "ancestor", "generation", "neighborhood", "community", "population"
]

DB_FILE = "leaderboard.json"
STATS_CACHE_FILE = "daily_stats.json"


def load_leaderboard():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            leaderboard = json.load(f)
        daily_records = [r for r in leaderboard if r.get("date") == today_str]
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(daily_records, f, ensure_ascii=False, indent=4)
        return daily_records
    except (json.JSONDecodeError, KeyError):
        return []


def save_score(username, date_str, attempts):
    leaderboard = load_leaderboard()
    leaderboard.append({"username": username, "date": date_str, "attempts": attempts})
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=4)


def display_daily_leaderboard(date_str):
    leaderboard = load_leaderboard()
    daily_records = [r for r in leaderboard if r["date"] == date_str]
    daily_records.sort(key=lambda x: x["attempts"])

    print("\n" + "🏆" * 15)
    print(f"  TODAY'S LEADERBOARD ({date_str})")
    print("🏆" * 15)
    print(f"{'Rank':<6}{'Player':<20}{'Total Guesses':<10}")
    print("-" * 40)

    if not daily_records:
        print("You are the first one to clear the game today!")
    else:
        for index, record in enumerate(daily_records, start=1):
            print(f"{index:<6}{record['username']:<20}{record['attempts']:<10}")
    print("-" * 40 + "\n")


def calculate_hybrid_similarity(word1, word2):
    w1_clean = word1.strip().lower()
    w2_clean = word2.strip().lower()
    if w1_clean == w2_clean:
        return 100.0

    token1 = nlp(w1_clean)
    token2 = nlp(w2_clean)
    if not token1.has_vector or not token2.has_vector:
        return 0.0

    semantic_score = token1.similarity(token2) * 100
    lexical_score = difflib.SequenceMatcher(None, w1_clean, w2_clean).ratio() * 100
    final_score = (semantic_score * 0.85) + (lexical_score * 0.15)
    final_score = round(final_score, 2)

    if final_score >= 99.0:
        return 99.00
    elif final_score < 0:
        return 0.00
    return final_score


def get_cached_target_stats(target_word, date_str):
    """
    Retrieves stats using ultra-fast matrix operations.
    Now also caches the actual word string of the 1st nearest word.
    """
    if os.path.exists(STATS_CACHE_FILE):
        try:
            with open(STATS_CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                if cache_data.get("date") == date_str and cache_data.get("target_word") == target_word:
                    # Return the 1st word's string along with scores
                    return cache_data["1st"], cache_data["10th"], cache_data["1000th"], cache_data.get("1st_word",
                                                                                                       "Unknown")
        except (json.JSONDecodeError, KeyError):
            pass

    print("Analyzing the global English vocabulary for daily stats...")

    target_vector = nlp(target_word).vector
    if np.all(target_vector == 0):
        return 0.0, 0.0, 0.0, "Unknown"

    vocab_vectors = nlp.vocab.vectors.data
    max_rows = vocab_vectors.shape[0]

    dot_product = np.dot(vocab_vectors, target_vector)
    vocab_norms = np.linalg.norm(vocab_vectors, axis=1)
    target_norm = np.linalg.norm(target_vector)

    vocab_norms[vocab_norms == 0] = 1e-10
    similarity_array = (dot_product / (vocab_norms * target_norm)) * 100

    # Store words and scores as tuples for proper tracking
    all_word_scores = []
    seen_words = set()

    for row_idx, word_hash in enumerate(nlp.vocab.vectors):
        if row_idx >= max_rows:
            break

        word_text = nlp.vocab.strings[word_hash]

        # Restrict the scope to valid alphabetical words with 2 or more letters
        if word_text.isalpha() and len(word_text) >= 2:
            word_str = word_text.lower().strip()
            if word_str != target_word and word_str not in seen_words:
                seen_words.add(word_str)
                all_word_scores.append((word_str, similarity_array[row_idx]))

    # Sort in descending order based on similarity scores (x[1])
    all_word_scores.sort(key=lambda x: x[1], reverse=True)

    nearest_1st_word = all_word_scores[0][0] if len(all_word_scores) >= 1 else "Unknown"
    nearest_1st = round(float(all_word_scores[0][1]), 2) if len(all_word_scores) >= 1 else 0.0
    nearest_10th = round(float(all_word_scores[9][1]), 2) if len(all_word_scores) >= 10 else 0.0
    nearest_1000th = round(float(all_word_scores[999][1]), 2) if len(all_word_scores) >= 1000 else (
        round(float(all_word_scores[-1][1]), 2) if all_word_scores else 0.0)

    nearest_1st = 99.99 if nearest_1st >= 100.0 else nearest_1st
    nearest_10th = 99.99 if nearest_10th >= 100.0 else nearest_10th

    cache_data = {
        "date": date_str,
        "target_word": target_word,
        "1st": nearest_1st,
        "1st_word": nearest_1st_word,  # Cache the actual text of the 1st nearest word
        "10th": nearest_10th,
        "1000th": nearest_1000th
    }
    with open(STATS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)

    return nearest_1st, nearest_10th, nearest_1000th, nearest_1st_word


def play_game():
    print("\n" + "=" * 50)
    print("      Welcome to Daily Semantic Semantle")
    print("=" * 50)
    username = input("Enter your username: ").strip()
    if not username:
        username = "Guest_" + str(random.randint(1000, 9999))

    today_str = datetime.date.today().strftime("%Y-%m-%d")

    random.seed(today_str)
    target_word = random.choice(TARGET_WORD_POOL)
    random.seed(None)

    # Retrieve the 1st word's text string from the stats loader
    nearest_1st, nearest_10th, nearest_1000th, nearest_1st_word = get_cached_target_stats(target_word, today_str)

    history = []
    attempts = 0

    print("\n" + "=" * 50)
    print(f" Hello, {username}! Let's guess Today's Secret Word.")
    print(f" Date: {today_str} | Target Pool: {len(TARGET_WORD_POOL)} curated words.")
    print(f" Semantic Hint: The nearest word in English vocabulary has a similarity of {nearest_1st},")
    print(f"                the 10th-nearest has a similarity of {nearest_10th},")
    print(f"                and the 1000th-nearest word has a similarity of {nearest_1000th}.")
    print(" Note: You can guess ANY valid English word recognized by our NLP system!")
    print(" Type 'exit' to give up.")
    print("=" * 50 + "\n")

    while True:
        guess = input("Enter a word: ").strip().lower()

        if guess == 'exit':
            print(f"\n🏳️ You gave up! Today's secret word was: **{target_word}**")
            # Reveal the 1st nearest word as a consolidation prize upon giving up
            break

        if not guess.isalpha():
            print("❌ Please enter a valid English word (letters only).")
            continue

        if not nlp(guess).has_vector:
            print("⚠️ This word is not recognized in our NLP dictionary. Try another word.")
            continue

        attempts += 1
        similarity_score = calculate_hybrid_similarity(guess, target_word)

        if guess == target_word:
            print("\n" + "*" * 50)
            print(f"🎉 CONGRATULATIONS! You found the word: **{target_word}**")
            print(f"📊 Your Total Attempts: {attempts}")
            # Surprise revelation of the 1st nearest word on game clearance
            print(
                f"💡 Fun Fact: The closest word to today's answer was **'{nearest_1st_word}'** with {nearest_1st} similarity!")
            print("*" * 50)

            save_score(username, today_str, attempts)
            display_daily_leaderboard(today_str)
            break

        if not any(h[0] == guess for h in history):
            history.append((guess, similarity_score))
            history.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "-" * 40)
        print(f"{'Rank':<6}{'Word':<20}{'Similarity':<10}")
        print("-" * 40)

        displayed = False
        for index, (h_word, h_score) in enumerate(history[:10], start=1):
            if h_word == guess:
                print(f"👉 {index:<4}{h_word:<20}{h_score:<10} (Current)")
                displayed = True
            else:
                print(f"   {index:<4}{h_word:<20}{h_score:<10}")

        if not displayed:
            current_rank = next(i for i, v in enumerate(history, start=1) if v[0] == guess)
            print("-" * 40)
            print(f"👉 {current_rank:<4}{guess:<20}{similarity_score:<10} (Current)")

        print("-" * 40)
        print(f"Total Guesses: {attempts}\n")


if __name__ == "__main__":
    # Automatically clear any outdated cache files upon startup for stable initial processing
    if os.path.exists(STATS_CACHE_FILE):
        os.remove(STATS_CACHE_FILE)
    play_game()
