import string
import mysql.connector
import nltk
import gensim.downloader as api
import json
from flask import Flask, redirect, render_template, request, url_for
from difflib import SequenceMatcher

app = Flask(__name__)

# Connect to the MySQL database using a connection pool
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Cognate_1234",
    "database": "transliteration",
    "pool_name": "cognate_pool",
    "pool_size": 5  # Adjust the pool size as needed
}
db = mysql.connector.pooling.MySQLConnectionPool(**db_config)

# Download NLTK data for WordNet if not available
nltk.download('wordnet')

malay_mapping = {
    'a': 'ا', 'b': 'ب', 'c': 'چ', 'd': 'د', 'e': 'ء', 'f': 'ف', 'g': 'ڬ', 'h': 'ه',
    'i': 'ي', 'j': 'ج', 'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن', 'o': 'و', 'p': 'ڤ',
    'q': 'ق', 'r': 'ر', 's': 'س', 't': 'ت', 'u': 'و', 'v': 'ڤ', 'w': 'و', 'x': 'خ',
    'y': 'ي', 'z': 'ز'
}

def transliterate_arabic_to_english(arabic_word):
    # The mapping for transliteration
    mapping = {
        'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
        'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
        'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
        'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y'
    }
    

    english_word = ''
    for char in arabic_word:
        if char in mapping:
            english_word += mapping[char]
        else:
            english_word += char
    return english_word

# Load the pre-trained FastText word embeddings
fasttext_model = api.load("fasttext-wiki-news-subwords-300")

# Cache for storing linguistic and semantic similarities
similarity_cache = {}

def calculate_linguistic_similarity(word1, word2):
    # Check if the similarity is cached
    cache_key = (word1, word2)
    if cache_key in similarity_cache:
        return similarity_cache[cache_key]

    word1_set = set(word1.lower())
    word2_set = set(word2.lower())

    intersection = word1_set.intersection(word2_set)
    union = word1_set.union(word2_set)

    similarity = len(intersection) / len(union)

    # Cache the similarity
    similarity_cache[cache_key] = similarity

    return similarity

def calculate_semantic_similarity(word1, word2):
    if word1 in fasttext_model.key_to_cognate_finder and word2 in fasttext_model.key_to_cognate_finder:
        vector1 = fasttext_model.get_vector(word1)
        vector2 = fasttext_model.get_vector(word2)
        similarity = fasttext_model.cosine_similarities(vector1, [vector2])[0]
        return similarity
    else:
        return 0.0  # Return 0 if one or both words are not present in the model's vocabulary

def transliterate_arabic_to_malay(arabic_word):
    malay_word = ''
    for char in arabic_word:
        if char in malay_mapping:
            malay_word += malay_mapping[char]
        else:
            malay_word += char
    return malay_word

def calculate_levenshtein_similarity(word1, word2):
    # Calculate Levenshtein distance
    from nltk.metrics.distance import edit_distance
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 0.0
    levenshtein_distance = edit_distance(word1, word2)
    levenshtein_similarity = 1.0 - (levenshtein_distance / max_len)
    return levenshtein_similarity

def calculate_string_similarity(word1, word2):
    # Use Levenshtein similarity
    similarity = calculate_levenshtein_similarity(word1, word2)
    return similarity

def calculate_cognate_percentage(string_similarity):
    # If you prefer, you can adjust the threshold here instead of hardcoding 0.5 (50%)
    threshold = 0.5
    return int(string_similarity * 100)

def transliterate_english_to_latin(english_word):
    # The mapping for transliteration
    mapping = {
        'a': 'a', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h',
        'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p',
        'q': 'k', 'r': 'r', 's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'ks',
        'y': 'y', 'z': 'z'
    }
    latin_word = ''
    for char in english_word:
        if char.lower() in mapping:
            latin_word += mapping[char.lower()]
        else:
            latin_word += char
    return latin_word

def search_word_in_database(word, is_arabic=True):
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        if is_arabic:
            query = "SELECT latin_word FROM words WHERE arabic_word = %s"
        else:
            query = "SELECT latin_word FROM english_words WHERE english_word = %s"
        cursor.execute(query, (word,))
        result = cursor.fetchone()
        return result[0] if result else None
    except mysql.connector.Error as e:
        print("Error while executing the database query:", e)
        return None
    finally:
        cursor.close()
        conn.close()


def convert_english_to_latin(english_word):
    database_result = search_word_in_database(english_word, is_arabic=False)
    if database_result:
        return database_result
    else:
        return transliterate_english_to_latin(english_word)


def search_word_in_database_english(english_word):
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        query = "SELECT latin_word FROM english_words WHERE english_word = %s"
        cursor.execute(query, (english_word,))
        result = cursor.fetchone()
        return result[0] if result else None
    except mysql.connector.Error as e:
        print("Error while executing the database query:", e)
        return None
    finally:
        cursor.close()
        conn.close()

def transliterate_malay_to_latin(malay_word):
    malay_mapping = {
        'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h',
        'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p',
        'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x',
        'y': 'y', 'z': 'z'
    }

    latin_word = ''
    for char in malay_word:
        if char in malay_mapping:
            latin_word += malay_mapping[char]
        else:
            latin_word += char
    return latin_word

def save_suggestion_to_database(arabic_word, english_word):
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        query = "INSERT INTO suggestions (arabic_word, english_word) VALUES (%s, %s)"
        cursor.execute(query, (arabic_word, english_word))
        conn.commit()
        print("Suggestion saved to the database.")
    except mysql.connector.Error as e:
        print("Error while saving suggestion to the database:", e)
    finally:
        cursor.close()
        conn.close()



@app.route('/en_my', methods=['GET', 'POST'])
def en_my():
    transliterated_malay_word = ''
    transliterated_english_word = ''
    cognate_percentage = None
    cognate_words = ""
    cognate_color = ""
    json_data_string = ""

    malay_word = ""
    english_word = ""

    if request.method == 'POST':
        malay_word = request.form['malay_word']  # Updated field name to 'malay_word'
        english_word = request.form['english_word']  # Field name remains 'english_word'

        if malay_word and english_word:
            # Search the Malay word in the database
            database_result = search_word_in_database(malay_word, is_arabic=False)
            if database_result:
                transliterated_malay_word = database_result
            else:
                transliterated_malay_word = transliterate_malay_to_latin(malay_word)

            # Search the English word in the database
            database_result = search_word_in_database_english(english_word)

            if database_result:
                transliterated_english_word = database_result
            else:
                transliterated_english_word = convert_english_to_latin(english_word)

            string_similarity = calculate_string_similarity(
                transliterated_malay_word.lower(), transliterated_english_word.lower())

            cognate_percentage = calculate_cognate_percentage(string_similarity)

            print(f"The similarity percentage is: {cognate_percentage}%")

    if cognate_percentage is None:
        cognate_percentage = 0

    if cognate_percentage >= 50:
        cognate_words = "Cognate words!"
        cognate_color = "green"
    else:
        cognate_words = "Not cognate words."
        cognate_color = "red"

    json_data = {
        "cognate_percentage": cognate_percentage,
        "transliterated_malay_word": transliterated_malay_word,
        "transliterated_english_word": transliterated_english_word
    }

    json_data_string = json.dumps(json_data)

    form_submitted = False

    if request.method == 'POST':
        form_submitted = True

    return render_template('en_my.html', json_data=json_data_string,
                           malay_word=malay_word, english_word=english_word,
                           form_submitted=form_submitted,
                           transliterated_malay_word=transliterated_malay_word,
                           transliterated_english_word=transliterated_english_word,
                           cognate_percentage=cognate_percentage,
                           cognate_words=cognate_words,
                           cognate_color=cognate_color,
                           jsonData=json_data_string)

# New Flask route for Malay-Arabic page
@app.route('/ar_my', methods=['GET', 'POST'])
def ar_my():
    transliterated_arabic_word = ''
    transliterated_malay_word = ''
    cognate_percentage = None
    cognate_words = ""
    cognate_color = ""
    json_data_string = ""

    arabic_word = ""
    malay_word = ""

    if request.method == 'POST':
        arabic_word = request.form['arabic_word']
        malay_word = request.form['malay_word']

        if arabic_word and malay_word:
            # Search the Malay word in the database
            database_result = search_word_in_database(malay_word, is_arabic=False)
            if database_result:
                transliterated_malay_word = database_result
            else:
                transliterated_malay_word = transliterate_malay_to_latin(malay_word)

            # Search the Arabic word in the database
            database_result = search_word_in_database(arabic_word)
            if database_result:
                transliterated_arabic_word = database_result
            else:
                transliterated_arabic_word = transliterate_arabic_to_english(arabic_word)

            string_similarity = calculate_string_similarity(
                transliterated_arabic_word.lower(), transliterated_malay_word.lower())

            cognate_percentage = calculate_cognate_percentage(string_similarity)

            print(f"The similarity percentage is: {cognate_percentage}%")

    if cognate_percentage is None:
        cognate_percentage = 0

    if cognate_percentage >= 50:
        cognate_words = "Cognate words!"
        cognate_color = "green"
    else:
        cognate_words = "Not cognate words."
        cognate_color = "red"

    json_data = {
        "cognate_percentage": cognate_percentage,
        "transliterated_arabic_word": transliterated_arabic_word,
        "transliterated_malay_word": transliterated_malay_word
    }

    json_data_string = json.dumps(json_data)

    form_submitted = False

    if request.method == 'POST':
        form_submitted = True

    return render_template('ar_my.html', json_data=json_data_string,
                           arabic_word=arabic_word, malay_word=malay_word,
                           form_submitted=form_submitted,
                           transliterated_arabic_word=transliterated_arabic_word,
                           transliterated_malay_word=transliterated_malay_word,
                           cognate_percentage=cognate_percentage,
                           cognate_words=cognate_words,
                           cognate_color=cognate_color,
                           jsonData=json_data_string)









@app.route('/cognate_finder', methods=['GET', 'POST'])
def cognate_finder():
    transliterated_arabic_word = ''
    transliterated_english_word = ''
    cognate_percentage = None
    cognate_words = ""
    cognate_color = ""
    json_data_string = ""  # Initialize the JSON data string with an empty string

    arabic_word = ""
    english_word = ""

    if request.method == 'POST':
        arabic_word = request.form['arabic_word']
        english_word = request.form['english_word']

        if arabic_word and english_word:
            # Search the Arabic word in the database
            database_result = search_word_in_database(arabic_word)
            if database_result:
                # Use the corresponding English word from the database
                transliterated_arabic_word = database_result
            else:
                # Transliterate the Arabic word
                transliterated_arabic_word = transliterate_arabic_to_english(arabic_word)

            # Search the English word in the database
            database_result = search_word_in_database_english(english_word)

            if database_result:
                # Use the corresponding Latin word from the database
                transliterated_english_word = database_result
            else:
                # Convert the English word to Latin
                transliterated_english_word = convert_english_to_latin(english_word)

            # Calculate string similarity between the English word and Transliterated Arabic word
            string_similarity = calculate_string_similarity(transliterated_arabic_word.lower(), transliterated_english_word.lower())

            # Calculate the cognate percentage
            cognate_percentage = calculate_cognate_percentage(string_similarity)

            # Print the percentage to the console
            print(f"The similarity percentage is: {cognate_percentage}%")

    # Ensure cognate_percentage is not None
    if cognate_percentage is None:
        cognate_percentage = 0

    if cognate_percentage >= 50:
        cognate_words = "Cognate words!"
        cognate_color = "green"
    else:
        cognate_words = "Not cognate words."
        cognate_color = "red"

# Prepare the JSON data to pass to JavaScript
    json_data = {
        "cognate_percentage": cognate_percentage,
        "transliterated_arabic_word": transliterated_arabic_word,
        "transliterated_english_word": transliterated_english_word
    }

            # Convert the JSON data to a JSON string
    json_data_string = json.dumps(json_data)

    form_submitted = False  # Initialize the flag
    
    if request.method == 'POST':
        form_submitted = True  # Set the flag if the form was submitted

    # Include cognate_percentage and JSON data in the template rendering
    return render_template('cognate_finder.html', json_data=json_data_string, arabic_word=arabic_word, english_word=english_word, form_submitted=form_submitted, transliterated_arabic_word=transliterated_arabic_word, transliterated_english_word=transliterated_english_word, cognate_percentage=cognate_percentage, cognate_words=cognate_words, cognate_color=cognate_color, jsonData= json_data_string)


@app.route('/suggest', methods=['GET', 'POST'])
def suggest_cognate():
    if request.method == 'POST':
        arabic_word = request.form['arabic_word']
        english_word = request.form['english_word']

        # Save the suggestion to the database (create a new table for suggestions)
        save_suggestion_to_database(arabic_word, english_word)

        message = "Thank you for your suggestion! It has been received."
        message_type = "success"

        return redirect(url_for('suggest_cognate'))
        # Optionally, you can redirect the user to a confirmation page
        return render_template('suggest.html', message=message, message_type=message_type)
    
    return render_template('suggest.html')

@app.route('/', methods=['GET'])

def landing_page():
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
