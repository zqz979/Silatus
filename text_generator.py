from abc import ABC, abstractmethod
from random import randint
from string import punctuation

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from json import load

# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
# pre-trained models available in 'spacy'
# It is trained on a large corpus of texts and it can be used to perform various nlp tasks such as part-of-speech tagging, 
# named entity recognition, and dependency parsing.
nlp = spacy.load('en_core_web_md')

def generate_sentence_prefix():
    all_synonyms = {
        'website': ['site', 'page', 'webpage', 'web page', 'internet site', 'Internet website', 'online page'],
        'create': ['make', 'build', 'construct', 'design', 'produce', 'generate', 'craft', 'synthesize', 'i want',
                   'we need', 'we want', 'give', ''],
        'me': ['for me', 'us', 'our team', 'my team', 'the team', 'the company', 'the business', 'our group', ''],
        'with': ['that contains', 'that has', 'having', 'made of', 'composed of', 'containing', 'where'],
    }
    sentence_base = ['Create', 'me', 'a', 'website', 'with']
    for i, word in enumerate(sentence_base):
        lower_word = word.lower()
        # Special Rule: Remove 'me' when 'Create' is not used
        if lower_word == 'me' and i - 1 >= 0 and (sentence_base[i - 1] == '' or len(sentence_base[i - 1].split()) > 1):
            # Remove 'me'
            sentence_base[i] = ''
            continue
        if lower_word in all_synonyms:
            word_synonyms = all_synonyms[lower_word]
            # Decide which word to use; we subtract 1 because the default word doesn't have an index
            word_to_use = randint(0, len(word_synonyms)) - 1
            if word_to_use > -1:
                sentence_base[i] = word_synonyms[word_to_use]

    # Capitalize first word of sentence, most of the time
    if sentence_base[0] and randint(0, 5) > 0: # changed because it is giving errors
        sentence_base[0] = sentence_base[0][0].upper() + sentence_base[0][1:]

    return ' '.join(sentence_base)


def get_readable_relative_position(position):
    text_position = ''

    # Corner positions
    if position['horizontal'] == -2 and position['vertical'] == -2:
        text_position += 'top left corner'
    elif position['horizontal'] == 2 and position['vertical'] == -2:
        text_position += 'top right corner'
    elif position['horizontal'] == -2 and position['vertical'] == 2:
        text_position += 'bottom left corner'
    elif position['horizontal'] == 2 and position['vertical'] == 2:
        text_position += 'bottom right corner'
    # Edge positions (non-corner)
    elif position['horizontal'] == -1 and position['vertical'] == -2:
        text_position += 'top and left center'
    elif position['horizontal'] == 0 and position['vertical'] == -2:
        text_position += 'top center'
    elif position['horizontal'] == 1 and position['vertical'] == -2:
        text_position += 'top and right center'
    elif position['horizontal'] == -1 and position['vertical'] == 2:
        text_position += 'bottom and left center'
    elif position['horizontal'] == 0 and position['vertical'] == 2:
        text_position += 'bottom center'
    elif position['horizontal'] == 1 and position['vertical'] == 2:
        text_position += 'bottom and right center'
    elif position['horizontal'] == -2 and position['vertical'] == -1:
        text_position += 'left edge and above center'
    elif position['horizontal'] == -2 and position['vertical'] == 0:
        text_position += 'left edge and vertical center'
    elif position['horizontal'] == -2 and position['vertical'] == 1:
        text_position += 'left edge and below center'
    elif position['horizontal'] == 2 and position['vertical'] == -1:
        text_position += 'right edge and above center'
    elif position['horizontal'] == 2 and position['vertical'] == 0:
        text_position += 'right edge and vertical center'
    elif position['horizontal'] == 2 and position['vertical'] == 1:
        text_position += 'right edge and below center'
    # center position
    elif position['horizontal'] == 0 and position['vertical'] == 0:
        text_position += 'center'
    return text_position


# The TextGenerator Abstract Base Class serves as the skeleton, or parent, for all text generation subclasses. On
# initialization, any subclass of this class will open and load the metadata from the specified file path
# (see example below).
class TextGenerator(ABC):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f: #fixed using utf-8
            self.metadata = load(f)

    @abstractmethod
    def generate(self):
        pass


# This text generator subclass uses nltk to remove proper nouns and nearby stop words from descriptions.
# You can create your own text generator subclasses and modify this one.
# Each subclass should have a specific function that is described in comments.
class NoProperNounsTextGenerator(TextGenerator):
    # From: https://www.geeksforgeeks.org/python-text-summarizer/
    def generate(self):
        # sanity checks
        if ('desc' not in self.metadata) or (self.metadata['desc'] == ""):
            return ''
        # Tokenizing the text
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(self.metadata['desc'])

        # Storing proper nouns so we can remove them
        proper_noun_indexes = []
        i = 0
        for (word, tag) in pos_tag(words):
            if tag == 'NNP' or tag == 'NNPS': # added plural proper nouns
                proper_noun_indexes.append(i)

            i += 1

        # Storing stop words that sit next to proper nouns so we can remove them
        stop_word_removal_indexes = []
        for idx in proper_noun_indexes:
            if idx < len(words) - 1 and words[idx + 1] in stopWords:
                stop_word_removal_indexes.append(idx + 1)
            if idx > 0 and words[idx - 1] in stopWords:
                stop_word_removal_indexes.append(idx - 1)

        removal_indexes = sorted(proper_noun_indexes + stop_word_removal_indexes, reverse=True)
        for idx in removal_indexes:
            words.pop(idx)

        stripped_text = ''.join([' ' + word if word not in punctuation else word for word in words])[1:]

        capitalized_sentences = ''
        for sentence in sent_tokenize(stripped_text):
            capitalized_sentences += sentence.capitalize() + ' '
        return capitalized_sentences[:-1]


# Create a subclass that describes the location of objects on the screen (i.e. buttons, images, inputs, iframes) and
# colors used.
class ObjectLocationTextGenerator(TextGenerator):
    def generate(self):
        sentence_prefix = generate_sentence_prefix()

        # Process images
        images = []
        if 'images' not in self.metadata:
            pass
        else:
            for image in self.metadata['images']:
                # Skip images that are not on screen
                if not image['is_displayed']:
                    continue
                images.append('an image of ' + image['alt'] + ' in the ' + get_readable_relative_position(image['position']))
        
        # process buttons
        buttons = []
        if 'buttons' not in self.metadata:
            pass
        else:
            for button in self.metadata['buttons']:
                # Skip images that are not on screen
                if not button['is_displayed']:
                    continue
                if 'alt' in button:
                    buttons.append('a button of ' + button['alt'] + ' in the ' + get_readable_relative_position(button['position']))
                else:
                    buttons.append('a button of ' + button['text'] + ' in the ' + get_readable_relative_position(button['position']))
        
        return images, buttons

# Create a subclass that describes navigation options (see "navbar" in the json).
class NavDescriptionTextGenerator(TextGenerator):
    def generate(self):
        # check if navbar is not present or empty
        if ('navbar' not in self.metadata) or self.metadata['navbar'] == '':
            return []
        # convert navbar options to list
        if '\n' in self.metadata['navbar']: # if \n appears, it is the delimiter 
            options = self.metadata['navbar'].split("\n")
        else: # otherwise a space is delimiter
            options = word_tokenize(self.metadata['navbar'])
        return options

# Create a subclass that summarizes the page based on text content and description metadata (and other metadata if it
# works well). Feel free to use any public NLP APIs, as long as they are free or very affordable (< $0.01 per 500
# words).
# using 'spacy' model
# md can be used to identify the most important sentences in the document based on their semantic similarity
# this can be achieved by computing a vector representation for each sentence in the document using the pre-trained word embeddings, and then
# clustering the sentences based on their vector similarity
# extract the sentences that are closest to the centroid of each cluster are considered to be the most representative sentences
# but still not accurate for part of the data (pending on fix, also the formatting)
class ContentSummaryTextGenerator(TextGenerator):
    # def __init__(self, text):
    #     self.metadata = {'text': text}
    #     self.nlp = spacy.load('en_core_web_md')
        
    # def generate(self):
    #     if 'text' not in self.metadata:
    #         return ''
    #     text = self.metadata['text']
    #     doc = self.nlp(text)
    #     sentences = [sent.text for sent in doc.sents]
        
    #     # Use TF-IDF vectorization to extract important phrases from the text
    #     vectorizer = TfidfVectorizer(stop_words='english')
    #     X = vectorizer.fit_transform(sentences)
    #     feature_names = np.array(vectorizer.get_feature_names())
    #     kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        
    #     # Identify the most important phrases based on the centroid of each cluster
    #     centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    #     important_phrases = []
    #     for i in range(3):
    #         cluster_indices = np.where(kmeans.labels_ == i)[0]
    #         important_phrases.append(feature_names[centroids[i, :5]])
        
    #     # Summarize the text by including the sentences that contain important phrases
    #     summary = [sentence for sentence in sentences if len(sentence.split()) > 5 and any(phrase in sentence.lower() for phrase in important_phrases)]
    #     return ' '.join(summary)
    def generate(self):
        if 'text' not in self.metadata:
            return ''

        doc = nlp(self.metadata['text'])
        output = []
        for sentence in doc.sents:
        # check if the sentence contains at least one non-stopword token
            if not any(token.is_stop == False for token in sentence):
                continue
            
        # check if the first token is a proper noun (capitalized)
            if not sentence[0].is_title:
                continue
        
        # check if the sentence contains any digits or capitalized words
            if any(token.is_digit or token.is_upper for token in sentence):
                continue
        
        # add the sentence text to the output
            output.append(str(sentence))
        
        return '\n'.join(output)
    

# BONUS: Create a subclass that combines elements from all other subclasses to generate text of variable length.
# Number of words need not be exact. A variance of up to 20% should be allowable
# (i.e. if words=50, generate(words=50) can return 40 to 60 words).
#class ContentSummaryTextGenerator(TextGenerator):


# This is an example of a text generator subclass instantiation. All text generator subclasses require the file path
# to metadata.

for i in range(10):
    desc_generator = ContentSummaryTextGenerator(f'./data/{str(i+1).zfill(4)}/metadata.json')
    print(f'Metadata {str(i+1).zfill(4)}: ', desc_generator.generate())

