# %% Import the necessary libraries
import os
import nltk
import pickle
import argparse
import numpy             as np
import matplotlib.pyplot as plt
import seaborn 			 as sns

# Set the random seed    
np.random.seed( 0 )

# Download the nltk corpus
nltk.download( 'reuters'   )

# "After" download, import the reuters and stopwords
# The details are contained in:
# [REF] https://www.nltk.org/book/ch02.html
from nltk.corpus import reuters

# Careful that scipy.special functions have accuracy issues
# [REF] https://stackoverflow.com/questions/21228076/the-precision-of-scipy-special-gammaln
from scipy.special import psi, polygamma, gammaln
import scipy.io

# %% Define stopwords not included as a vocabularies

stops = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "did", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "lt", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "said", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby", 
    '"(', '"...', ')"', '),"', ')-&', ').', ')...', ')>', ')>,', ',"', ',,', '--', '."', ".'", '.)', '?"', 
    '...', './', '.>', '0p', '11p', '14th', '17p', '1970s', '1990s', '1st', '20th', '242p', '25p', '25th', '2nd', '2p', 
    '317p', '3rd', '3x', '459p', '479p', '480p', '4th', '646p', '670p', '696p', '7p', '813p', '830p', '9p', ">'", '>.', '?"'
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y", "z", "inc", 
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", "),", ">,", "):" , ".", "!", "?", ",", ";", ":", "[", "]", "{", "}", "-", "+", 
    "_", "/", "@", "#", "$", "%", "^", "&", "*", "(", ")", "<", ">", "|", "=", ')...', '"...', ".'", '.)',
    ".-", ".,", "'", '"', ',",'
]

# %% Read the training set and test set
def create_vocab( n = 500 ):

    # Distinguish the id_set, training/test set of the Reuter's data set
    id_set, trainset = [ ], [ ]

    # The vocabulary list, produced from the training set 
    vocab = [ ]

    # Counter
    i = 0

    # Read only the training set of the data
    for file_id in reuters.fileids( ):

        if file_id.startswith( "train" ):

            # Save the words, except the (1) stopwords (2) numbers
            doc = [ w.lower( ) for w in reuters.words( file_id ) if ( w.lower( ) not in stops ) and ( not w.isnumeric( ) ) ]

            # Save the values
            if doc:
                trainset.append( doc )
                id_set.append( file_id )
                vocab += doc
                i += 1

        if i >= n:
            break


    # Within the vocabulary list, get only the unique vocabularies.
    vocab = list( set( vocab ) )
    vocab.sort( )
    
    return vocab, trainset, id_set

# %%
def _init_gibbs( docs, vocab, n_topic, n_gibbs = 2000 ):
    """
        Initialize t=0 state for Gibbs sampling.
        Replace initial word-topic assignment ndarray (M, N, N_GIBBS) in-place.
    """
    # initialize variables
    init_lda(docs, vocab, n_topic=n_topic, gibbs=True)
    
    # word-topic assignment
    N_max = max( N )
    assign = np.zeros( ( M, N_max, n_gibbs + 1) , dtype = int )
    print( f"assign: dim {assign.shape}" )
    
    # initial assignment
    for d in range( M ):
        for n in range( N[ d ] ):
            # randomly assign topic to word w_{dn}
            w_dn = docs[ d ][ n ]
            
            assign[d, n, 0] = np.random.randint( k )

            # increment counters
            i = assign[ d, n, 0 ]
            n_iw[ i, w_dn ] += 1
            n_di[ d, i    ] += 1
            
    return assign
            
def _conditional_prob( w_dn, d ):
    """
        P(z_{dn}^i=1 | z_{(-dn)}, w)
    """
    prob = np.empty( k )
    
    for i in range( k ):
        
        # P(w_dn | z_i)
        _1 = (n_iw[i, w_dn] + eta) / (n_iw[i, :].sum() + V*eta)
        # P(z_i | d)
        _2 = (n_di[d, i] + alpha) / (n_di[d, :].sum() + k*alpha)
        
        prob[i] = _1 * _2
    
    return prob / prob.sum()

def run_gibbs( docs, vocab, n_topic, n_gibbs=2000, verbose = True):
    """
    Run collapsed Gibbs sampling
    """
    # initialize required variables
    _init_gibbs(docs, vocab, n_topic, n_gibbs)
    
    if verbose:
        print( "\n", "="*10, "START SAMPLER", "="*10 )
    
    # run the sampler
    for t in range( n_gibbs ):
        for d in range( M ):
            for n in range( N[ d ] ):
                w_dn = docs[ d ][ n ]
                
                # decrement counters
                i_t = assign[d, n, t]  # previous assignment
                n_iw[ i_t, w_dn ] -= 1
                n_di[   d, i_t  ] -= 1

                # assign new topics
                prob = _conditional_prob( w_dn, d )
                i_tp1 = np.argmax( np.random.multinomial( 1, prob ) )

                # increment counter according to new assignment
                n_iw[ i_tp1, w_dn   ] += 1
                n_di[     d, i_tp1  ] += 1
                assign[   d, n, t+1 ] = i_tp1
        
        # print out status
        if verbose & ( ( t+1 ) % 50 == 0 ):
            print(f"Sampled {t+1}/{n_gibbs}")			
            

beta  = np.empty( ( k, V ) )
theta = np.empty( ( M, k ) )

for j in range( V ):
    for i in range( k ):
        beta[ i, j ] = ( n_iw[ i, j ] + eta ) / ( n_iw[ i, : ].sum() + V * eta )

for d in range( M ):
    for i in range( k ):
        theta[ d, i ] = ( n_di[ d, i ] + alpha ) / (n_di[ d, : ].sum() + k * alpha )			

if __name__ == "__main__":


    # Get argument parser to automize the script
    parser = argparse.ArgumentParser( )
    parser.add_argument( '--n_topics'  , type = int, default = 3   )   # The number of topics
    args = parser.parse_args( )

    k = args.n_topics
