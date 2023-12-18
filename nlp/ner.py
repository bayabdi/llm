import nltk
import pprint

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')


from nltk import word_tokenize, pos_tag, ne_chunk

text = """ Hello, Bael. This is Andrei """


def get_entities(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Perform part-of-speech tagging
    tagged_tokens = pos_tag(tokens)

    # Perform named entity recognition
    ne_tree = ne_chunk(tagged_tokens)

    # Extract named entities
    named_entities = []
    for subtree in ne_tree:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([token for token, pos in subtree.leaves()])
            entity_type = subtree.label()
            named_entities.append((entity, entity_type))

    return named_entities

def get_persons(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Perform part-of-speech tagging
    tagged_tokens = pos_tag(tokens)
    
    # Perform named entity recognition
    ne_tree = ne_chunk(tagged_tokens)
    
    persons = []
    
    # Extract persons from the named entities tree
    for subtree in ne_tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'PERSON':
            persons.append(' '.join([token[0] for token in subtree.leaves()]))
    
    return persons

pprint.pprint(get_entities(text))