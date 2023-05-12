# %% Import the necessary libraries
import os
import nltk
import pickle
import argparse
import numpy             as np


# Set the random seed    
np.random.seed( 0 )

# Download the nltk corpus
nltk.download( 'reuters'  )

# "After" download, import the reuters and stopwords
# The details are contained in:
# [REF] https://www.nltk.org/book/ch02.html
from nltk.corpus import reuters
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

# %% Function for running conditional probability 
def cond_prob( w_dn, d, k, n_iw, n_di, alpha, eta ):

    prob = np.empty( k )

    # Iterate over the topics k    
    for i in range( k ):

        # P(w_dn | z_i)
        # Equation 6, Finding Scientific Topics (2004)
        # Summation over the words
        tmp1 = ( n_iw[ i, w_dn ] + eta   ) / ( n_iw[ i, : ].sum( ) + V * eta   )

        # P(z_i | d)
        # Equation 7, Finding Scientific Topics (2004)
        # Summation over the topics
        tmp2 = ( n_di[ d, i    ] + alpha ) / ( n_di[ d, : ].sum( ) + k * alpha )
        
        # Equation 5, Finding Scientific Topics (2004)        
        prob[ i ] = tmp1 * tmp2
    

    return prob / prob.sum( )



# %% The Main Loop

def n_most_important( ttmp, vocab, n = 30 ):

    # Get beta with top 9 values
    max_values = ttmp.argsort( )[ -n : ][ ::-1 ]
    return np.array( vocab )[ max_values ]

if __name__ == "__main__":

    # Get argument parser to automize the script
    parser = argparse.ArgumentParser( )
    parser.add_argument( '--run_train' , action='store_true'        )   # To train the LDA model
    parser.add_argument( '--plot'      , action='store_true'        )   # To plot  the LDA model
    parser.add_argument( '--n_train'   , type = int, default = 700  )   # The number of reuters' documents used for the training
    parser.add_argument( '--n_topics'  , type = int, default = 3    )   # The number of topics
    parser.add_argument( '--n_samples' , type = int, default = 2000 )   # The number of reuters' documents used for the training

    args = parser.parse_args( )

    # The number of vocabularies in the dictionary
    # As with the LDA paper, we use "V" for the number of "words" (vocabs)
    ntrain   = args.n_train
    is_train = args.run_train
    is_plot  = args.plot
    k        = args.n_topics

    # There are already parsed vocabs from n_train articles
    tmp_name = "vocabs/vocab" + str( ntrain ) + ".pkl"

    # If the pickle file doesn't exists
    if not os.path.exists( tmp_name ):
        vocab, docs_word, id_set = create_vocab( ntrain )

        # Save zip
        vocab_zip = [ vocab, docs_word, id_set ]

        with open( tmp_name, "wb" ) as f:
            pickle.dump( vocab_zip, f, protocol = pickle.HIGHEST_PROTOCOL )

        print( tmp_name + " saved" )   

        # And regardless of the argument input, if pickle file doesn't exist you need to train the robot
        args.run_train = True

    # If the pickle file exists
    else:
        f = open( tmp_name , 'rb')
        vocab, docs_word, id_set = pickle.load( f )
        print( "loaded " + tmp_name )    

    # Initial Parameters
    V = len( vocab )

    # Produce a dictionary, with key-value pair as word-index (word_idx) pair.
    word_idx = { w : i for i, w in enumerate( vocab ) }
    idx_word = { i : w for i, w in enumerate( vocab ) }

    # Change the trainset/testset documents from a list of words to a list of index
    docs = [ np.array( [ word_idx.get( w, V ) for w in doc ] ) for doc in docs_word ] 
    
    # The numpy array of the length of each document
    N = np.array( [ len( doc ) for doc in docs ] )

    # The number of documents for the train set. 
    # Note that this should be identical to ntrain
    M = len( N )

    # The number of Gibbs sampling 
    n_gibbs = args.n_samples

    # The alpha and eta (beta for Eq. 6) for the Gibbs Sampling
    # Once alpha and eta are determined, it is not modified
    # alpha and eta are hyperparameters for the symmetric Dirichlet Prior
    alpha = np.random.gamma( shape = 100, scale = 0.01, size = 1 )  # one for all k
    eta   = np.random.gamma( shape = 100, scale = 0.01, size = 1 )  # one for all V

    # Equation 2, needs the # of topic vs. words, defining an integer matrix
    # The n_iw, where i is the topic, w is the vocab
    n_iw = np.zeros( ( k, V ) , dtype = int )

    # Equation 2, needs the # of document vs. topic, defining an integer matrix
    # The n_di, where d is the document, k is the topic
    n_di = np.zeros( ( M, k ) , dtype = int )

    # Initilize the Gibbs Sampling Method
    # The Maximum Number of Documents
    N_max = max( N )
    assign = np.zeros( ( M, N_max, n_gibbs + 1 ), dtype = int )
    print( f"assign: dim { assign.shape }" )
    
    # Initial assignment
    # Iterating through the documents.
    for d in range( M ):
        for n in range( N[ d ] ):

            # Randomly assign topic to word w_{dn}
            w_dn = docs[ d ][ n ]

            # Assign a random topic for the first trial of the d-th document, n-th word
            assign[ d, n, 0 ] = np.random.randint( k )

            # Get the topic of the d-th document's n-th word
            i = assign[ d, n, 0 ]
            n_iw[ i, w_dn ] += 1
            n_di[ d, i    ] += 1

    # If train, then
    if is_train:
        # Run Gibbs Sampling, the Main Loop
        for t in range( n_gibbs ):
            
            # Iterate through each words
            for d in range( M ):
                for n in range( N[ d ] ):

                    # Get the n-th word of the d-th document
                    w_dn = docs[ d ][ n ]
                    
                    # Get the current assignment
                    i_t = assign[ d, n, t ] 

                    # This is for n_{-i,j}^{wi} in Eq. 2
                    n_iw[ i_t, w_dn ] -= 1

                    # This is for n_{-i,j}^{di} in Eq. 3
                    n_di[   d, i_t  ] -= 1

                    # Get the Probability Distribution
                    prob  = cond_prob( w_dn, d, k, n_iw, n_di, alpha, eta )

                    # Get a single trial of the multinomial probability, which is simply a categorical
                    # np.argmax simply gets the index which outputs 1
                    i_tp1 = np.argmax( np.random.multinomial( 1, prob ) )

                    # increment counter according to new assignment
                    n_iw[  i_tp1,  w_dn ] += 1
                    n_di[      d, i_tp1 ] += 1

                    assign[  d, n, t+1 ] = i_tp1
            
            # print out status
            if ( (t+1) % 2 == 0 ):
                print(f"Sampled {t+1}/{n_gibbs}")

        # Recover beta and theta
        beta  = np.empty( ( k, V ) )
        theta = np.empty( ( M, k ) )

        for j in range( V ):
            for i in range( k ):
                beta[ i, j ]  =   ( n_iw[ i, j ] + eta ) / ( n_iw[ i, : ].sum( ) + V * eta   )

        for d in range( M ):
            for i in range( k ):
                theta[ d, i ] = ( n_di[ d, i ] + alpha ) / ( n_di[ d, : ].sum( ) + k * alpha )

        # Print out the topic of the d-th document
        for i in range( k ):
            tmp  = beta[ i, : ]
            print( f"TOPIC {i+1:02}: { n_most_important( tmp, vocab, 20 ) }")            

        # Once the training is complete, save the alpha, beta, gamma, phi, and other variables 
        scipy.io.savemat( "tmp/trained_v" + str( ntrain ) + "_" + str( n_gibbs ) + "_LDA_Gibbs.mat", { "beta": beta, "theta":theta, "n_iw": n_iw, "n_di":n_di, "M": M, "N": N, "V": V, "ntrain": ntrain } )

    else:

        # Load the data file
        # Code should be set before hand
        # file_name = "dataset/set1/trained_v" + str( ntrain ) + "_LDA_Gibbs1.mat"
        file_name = "dataset/set4/trained_v700_50_LDA_Gibbs.mat"
        data = scipy.io.loadmat( file_name )
        print( "data loaded" )

        # Print out the topic of the d-th document
        for i in range( k ):
            tmp  = data[ "beta" ][ i, : ]
            print( f"TOPIC {i+1:02}: { n_most_important( tmp, vocab, 20 ) }")