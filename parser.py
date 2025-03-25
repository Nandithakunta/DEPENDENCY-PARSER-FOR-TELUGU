import os
import subprocess
import pandas as pd
import numpy as np
import gradio as gr
from deep_translator import GoogleTranslator
import stanza
from spacy import displacy
import speech_recognition as sr

def read_conll(file_path):
    """Read CoNLL-formatted data."""
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                sentence.append(line.strip().split('\t'))
    return sentences

def write_conll(sentences, output_path):
    """Save data in CoNLL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token in sentence:
                f.write('\t'.join(token) + '\n')
            f.write('\n')

def display_conll(sentences, num_sentences=3):
    """Display CoNLL data for debugging."""
    for sentence in sentences[:num_sentences]:
        for token in sentence:
            print('\t'.join(token))
        print()

def train_parser(maltparser_jar, train_data, model_name):
    """Train the parser using MaltParser."""
    if not os.path.exists(maltparser_jar):
        print(f"Error: MaltParser JAR file not found at {maltparser_jar}")
        return
    if not os.path.exists(train_data):
        print(f"Error: Training data file not found at {train_data}")
        return

    print(f"Using MaltParser JAR: {maltparser_jar}")
    print(f"Training data file: {train_data}")
    print(f"Model name: {model_name}")

    command = [
        "java", "-jar", maltparser_jar,
        "-c", model_name,
        "-i", train_data,
        "-m", "learn"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Model '{model_name}' trained successfully.")
    else:
        print("Error during training:", result.stderr)

def test_parser(maltparser_jar, test_data, model_name, output_file):
    """Test the parser using MaltParser."""
    if not os.path.exists(maltparser_jar):
        print(f"Error: MaltParser JAR file not found at {maltparser_jar}")
        return
    if not os.path.exists(test_data):
        print(f"Error: Test data file not found at {test_data}")
        return

    print(f"Testing using model: {model_name}")

    command = [
        "java", "-jar", maltparser_jar,
        "-c", model_name,
        "-i", test_data,
        "-o", output_file,
        "-m", "parse"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Parsing completed. Output saved to: {output_file}")
    else:
        print("Error during parsing:", result.stderr)

if __name__ == "__main__":
    # Define paths
    maltparser_jar = "C:/Users/User/Downloads/maltparser-1.9.2.jar"  # Update this path
    train_data = "C:/Users/User/Downloads/te_mtg-ud-train.conllu"  # Update this path
    test_data = "C:/Users/User/Downloads/te_mtg-ud-test.conllu"  # Update this path
    output_file = "C:/Users/User/Downloads/parsed_output.conllu"  # Update this path
    model_name = "telugu_model"

    # Read and preprocess data
    sentences = read_conll(train_data)
    print("Sample Sentences (Original Data):")
    display_conll(sentences, num_sentences=2)

    # Train the parser
    train_parser(maltparser_jar, train_data, model_name)

    # Test the parser
    test_parser(maltparser_jar, test_data, model_name, output_file)

    # Display parsed output
    parsed_sentences = read_conll(output_file)
    print("Sample Sentences (Parsed Data):")
    display_conll(parsed_sentences, num_sentences=2)


# Download and load Stanza Telugu model
stanza.download('te')
nlp_te = stanza.Pipeline('te')

# Grammar Rules for Explanations
GRAMMAR_RULES = {
    "NOUN": "A noun represents a person, place, thing, or idea.",
    "VERB": "A verb indicates an action, occurrence, or state of being.",
    "ADJ": "An adjective describes or modifies a noun.",
    "ADV": "An adverb modifies a verb, an adjective, or another adverb.",
    "PRON": "A pronoun replaces a noun in a sentence.",
    "DET": "A determiner specifies or quantifies a noun.",
    "ADP": "An adposition is a preposition or postposition.",
    "CONJ": "A conjunction connects words, phrases, or clauses.",
    "PART": "A particle adds grammatical meaning but does not belong to a specific part of speech.",
    "INTJ": "An interjection expresses emotion or exclamation.",
    "NUM": "A numeral denotes a number.",
    "PUNCT": "Punctuation marks are symbols used to structure sentences.",
    "SYM": "A symbol represents special characters or mathematical operators.",
    "X": "Other category for unrecognized or foreign words.",
    "root": "The main word of the sentence, often the main verb.",
    "nsubj": "Nominal subject of the sentence.",
    "obj": "The object of a verb.",
    "iobj": "Indirect object of a verb.",
    "obl": "Oblique nominal, often an adverbial modifier.",
    "nmod": "Nominal modifier, providing additional information about a noun.",
    "advmod": "Adverbial modifier, describing the verb.",
    "amod": "Adjectival modifier, describing the noun.",
    "det": "Determiner of a noun phrase.",
    "case": "Case marker, like postpositions in Telugu.",
    "cc": "Coordinating conjunction connecting two elements.",
    "conj": "Conjunct, part of a conjunction.",
    "mark": "Marker, introducing a subordinate clause.",
    "compound": "Compound word relation.",
    "appos": "Appositional modifier, renaming another element.",
    "vocative": "Direct address to someone.",
    "discourse": "Discourse elements like interjections.",
    "punct": "Punctuation mark relation."
}

# Function to explain grammar
def explain_grammar(token):
    pos_rule = GRAMMAR_RULES.get(token['tag'], "No explanation available.")
    dep_rule = GRAMMAR_RULES.get(token.get('label', ''), "No explanation available.")
    return f"<strong>POS:</strong> {pos_rule}<br><strong>Dependency:</strong> {dep_rule}"

# Function to transcribe Telugu audio to text
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language="te-IN")
        except sr.UnknownValueError:
            return "Error: Unable to recognize the audio."
        except sr.RequestError as e:
            return f"Error: {str(e)}"

# Process input and generate analysis
def process_input(selected_sentence, typed_sentence, audio_file):
    if audio_file is not None:
        input_text = transcribe_audio(audio_file)
    else:
        input_text = selected_sentence if selected_sentence else typed_sentence

    try:
        # Translate Telugu sentence to English
        translator = GoogleTranslator(source='te', target='en')
        translated_sentence = translator.translate(input_text)

        # Parse Telugu sentence using Stanza
        doc = nlp_te(input_text)

        words = []
        arcs = []
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append({"text": word.text, "tag": word.upos, "label": word.deprel})
                if word.head > 0:
                    start = word.head - 1
                    end = sentence.words.index(word)
                    arcs.append({
                        "start": min(start, end),
                        "end": max(start, end),
                        "label": word.deprel,
                        "dir": "right" if start < end else "left",
                    })

        sentence_data = {"words": words, "arcs": arcs}
        html_dependency_tree = displacy.render(sentence_data, style="dep", manual=True, page=False)

        pos_tags_html = (
            "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
            "<tr><th>Word</th><th>POS Tag</th><th>Dependency</th></tr>"
            + "".join(
                f"<tr><td>{word['text']}</td><td>{word['tag']}</td><td>{word['label']}</td></tr>"
                for word in words
            )
            + "</table>"
        )

        grammar_html = "<h3>Grammar Explanations</h3>"
        for word in words:
            grammar_html += f"<p>{word['text']} - {explain_grammar(word)}</p>"

        return translated_sentence, html_dependency_tree, pos_tags_html, grammar_html

    except Exception as e:
        return f"Error: {str(e)}", "No visualization due to error.", "No POS tags due to error.", "No grammar explanations due to error."

# Load CoNLL data
def load_conll_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                columns = line.split('\t')
                if len(columns) > 1:
                    word = columns[1]
                    sentence.append(word)
            else:
                if sentence:
                    sentences.append(" ".join(sentence))
                    sentence = []
    if sentence:
        sentences.append(" ".join(sentence))
    return sentences

# Paths for dataset and MaltParser
maltparser_jar = "C:/Users/User/Downloads/maltparser-1.9.2.jar"
dataset_path = "C:/Users/User/OneDrive/Desktop/dependecncy/te_mtg-ud-train.conllu"
preloaded_sentences = load_conll_data(dataset_path)

# Define Gradio interface
interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Dropdown(
            choices=preloaded_sentences,
            label="Select a Telugu Sentence"
        ),
        gr.Textbox(
            lines=6,
            placeholder="Or type a Telugu sentence here...",
            label="Input Telugu Sentence"
        ),
        gr.Audio(
            type="filepath",
            label="Upload Telugu Speech (WAV Format)"
        )
    ],
    outputs=[
        gr.Textbox(label="🌐 Translated Sentence (English)"),
        gr.HTML(label="🔄 Dependency Tree Visualization"),
        gr.HTML(label="📜 Parts of Speech Tags (Table)"),
        gr.HTML(label="📖 Grammar Explanations")
    ],
    title="Dependency  parser for telugu",
    description="Translate, visualize, and analyze Telugu sentences using text or speech input.",
    theme="default",
    css=""" 
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; }
        .output-html {
            border: 2px solid #4CAF50;
            padding: 20px;
            margin: 24px 0;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h3 { color: #4CAF50; font-weight: bold; }
        table { margin-top: 20px; width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
    """
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(share=True, inbrowser=True)

