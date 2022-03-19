from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr
from collections import Counter

def get_freqs(corpus, puncts):
    freqs = {}
    ### BEGIN SOLUTION
    
    punc = puncts + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    words = []
    for char in corpus.lower():
        if char not in punc:
            words.append(char)
        else:
            words.append(' ')
        
    lcorpus = ''.join(words)
            
    freqs = dict(Counter(lcorpus.split()))
        
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    ### BEGIN SOLUTION
    
    sorted_freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))
    top_10 = list(sorted_freqs.keys())[:10]
    
    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    
    sorted_freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=False))
    bottom_10 = list(sorted_freqs.keys())[:10]
    
    ### END SOLUTION
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    
    singletons = 0
    for word in freqs:
        if freqs[word] == 1:
            singletons += 1
    
    perc = (singletons * 1.0 / len(freqs)) * 100
    
    return perc

    ### END SOLUTION

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    
    freqs = {}
    punc = puncts + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    words = []
    for char in corpus.lower():
        if char not in punc:
            words.append(char)
        else:
            words.append(' ')
        
    lcorpus = ''.join(words)
    
    words = lcorpus.split()
    stem_words = []
    
    porter = PorterStemmer()
    for word in words:
        stem_words.append(porter.stem(word))
            
    freqs = dict(Counter(stem_words))
    
    return freqs
    
    ### END SOLUTION

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    
    freqs = {}
    punc = puncts + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    words = []
    for char in corpus.lower():
        if char not in punc:
            words.append(char)
        else:
            words.append(' ')
        
    lcorpus = ''.join(words)
    
    words = lcorpus.split()
    lemm_words = []
    
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in words:
        lemm_words.append(wordnet_lemmatizer.lemmatize(word, pos='v'))
            
    freqs = dict(Counter(lemm_words))
    
    return freqs
    
    ### END SOLUTION

def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION
    
    return len(list(freqs.keys()))

    ### END SOLUTION

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    
    return len(list(freqs_stemming.keys()))

    ### END SOLUTION

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
    ### BEGIN SOLUTION
    
    return len(list(freqs_lemmatized.keys()))

    ### END SOLUTION

def percentage_of_unseen_vocab(a, b, length_i):
    ### BEGIN SOLUTION
    
    return len(set(a) - set(b)) * 1.0 / length_i

    ### END SOLUTION

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    
    sorted_freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))
    
    freq_count = 0
    word_count = 0
    total_count = 0
    total_words = len(sorted_freqs)
    
    for word in sorted_freqs:
        total_count += sorted_freqs[word]
    
    for word in sorted_freqs:
        freq_count += sorted_freqs[word]
        word_count += 1
        
        if (freq_count * 1.0 / total_count) >= 0.8:
            break
    
    return (word_count * 1.0) / total_words
    
    ### END SOLUTION

def plot_zipf(freqs):
    ### BEGIN SOLUTION
    
    sorted_freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))
    
    y = list(sorted_freqs.values())
    x = list(range(1, len(y) + 1))
    
    plt.plot(x, y)
    plt.xlabel('Rank of Word')
    plt.ylabel('Frequency of Word')
    
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

def get_TTRs(languages):
    TTRs = {}
    for lang in languages:
        words = udhr.words(lang)
        ### BEGIN SOLUTION
        
        lwords = [word.lower() for word in words]
        
        type_count = []
        for token_count in range(100, 1400, 100):
            unique_types = len(set(lwords[:token_count]))
            type_count.append(unique_types)
        
        TTRs[lang] = type_count
        
        ### END SOLUTION
    return TTRs

def plot_TTRs(TTRs):
    ### BEGIN SOLUTION
    
    x = list(range(100, 1400, 100))
    
    for lang in list(TTRs.keys()):
        plt.plot(x, TTRs[lang], label=lang)
        
    plt.xlabel('Total Words in Corpus')
    plt.ylabel('Unique Types in Corpus')
    plt.legend()
    
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.
