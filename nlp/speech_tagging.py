import nltk
import pprint

nltk.data.path.append("D:/model/nltk_data")

# nltk.download("punkt")
# nltk.download('averaged_perceptron_tagger')

text = "Best film of this year is Openheimer and  My name is Bayel!"

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Perform part of speech tagging on the tokenized words
tagged_words = nltk.pos_tag(tokens)

pprint.pprint(tagged_words)
