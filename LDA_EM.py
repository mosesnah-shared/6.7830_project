# %% Import the necessary libraries
import os
import time
import nltk
import pickle
import argparse
import numpy             as np
import matplotlib.pyplot as plt
import seaborn 			 as sns

# Set the random seed    
np.random.seed( 0 )

# Download the nltk corpus
nltk.download( 'reuters'  )

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


# %% Define the E-step of the EM algorithm
# Code modified from 
# https://github.com/naturale0/NLP-Do-It-Yourself/blob/main/NLP_with_PyTorch/3_document-embedding/3-1.%20latent%20dirichlet%20allocation.ipynb
def E_step( docs, phi, gamma, alpha, beta ):
    """
        Parameters
        ----------
            docs: the M documents 
                  The words of each document are indexed via the vocab. set.

            phi:  M x max( N ) x k array
                Summation along k topics (i.e., axis = 2) must be 1

            Gamma: M x k array

            alpha: k-length array            

            beta: k x V array
                Summation along k topics (i.e., axis = 9) must be 1
                
    """

    # Create a 3D beta array, to a size of M x max( N ) x k
    # This is for simplifying the calculation of phi_{dni}
    beta3D = np.zeros( ( M, max( N ), k ) )

    # Iterating over the documents
    for d in range( M ):
        beta3D[ d, :N[ d ], : ] = np.copy( beta[ :, docs[ d ] ].T )

    # Create Gamma, also to a size of M x max( N ) x k
    # "keepdims" save the dimension of the multi-dimensional array after summation.
    # https://stackoverflow.com/questions/41752442/dividing-3d-array-by-2d-row-sums
    tmp = np.exp( psi( gamma ) - psi( gamma.sum( axis = 1, keepdims=True ) ) )

    # Expand this array to axis 1 with max( N )
    gamma3D = np.stack( [ tmp for _ in range( max( N ) ) ], axis = 1 )

    # Update phi, based on the equation
    # Point-wise multiplication, which is way faster than others.
    phi = beta3D * gamma3D

    # Normalization of the phi
    # [REF] https://stackoverflow.com/questions/41752442/dividing-3d-array-by-2d-row-sums
    # [REF] https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    phi_sum = phi.sum( axis = 2, keepdims = True )
    phi = np.divide( phi, phi_sum, out = np.zeros_like( phi ), where=( phi_sum!=0 ) )
    
    # Update Gamma
    gamma = np.tile( alpha, ( M, 1 ) ) + phi.sum( axis = 1 )

    return phi, gamma


# %% Define the M-step of the EM algorithm
def M_step( docs, phi, gamma, alpha, beta, M ):

    # ======================================== #
    # ============= Update alpha ============= #
    # ======================================== #
    # Using Gradient Ascent

    # Define the maximum iteration and the tolerance to check convergence
    max_iter = int( 1e3 )
    tol      = 1e-6

    # Conduct iterations
    for _ in range( max_iter ):
        
        # Store old alpha value
        alpha_old = alpha.copy( )
        
        # g: gradient 
        g = M * ( psi( alpha.sum( ) ) - psi( alpha ) ) + ( psi( gamma ) - psi( gamma.sum( axis = 1, keepdims = True ) ) ).sum( axis = 0 )

        # z: Hessian constant component
        # NOT THE TOPIC z!!
        z =  M * polygamma( 1, alpha.sum( ) )  

        # h: Hessian diagonal component
        h = -M * polygamma( 1, alpha )     

        # The c value for the gradient update
        c = ( g / h ).sum( ) / ( 1./z + ( 1./h ).sum( ) )

        # update the alpha via gradient descent
        alpha -= (g - c) / h
        
        # check the error between old and new alphas
        err = np.sqrt( np.mean( ( alpha - alpha_old ) ** 2 ) )

        # Check the tolerance of the error
        if err < tol:
            break
    
    # ======================================== #
    # ============= Update beta ============== #
    # ======================================== #

    # update beta
    # phi      shape is M x max( N ) x k
    # w_dn^{j} shape is M x max( N ) x V

    # First, create w_nd{ j } array, and we use the following trick
    # Construct a M x max( N ) x V matrix, where the [:, :, i] element is filled with i

    # Construct 0,1,2...,V
    tmp  = np.arange( V )

    # Expand this to 3D array
    tmp2 = np.transpose( np.tile( tmp[ :, np.newaxis, np.newaxis], ( 1, M, max( N ) ) ), ( 1,2, 0 ) )
    
    # Create the matrix for masking
    # Copy the documents along the document axis
    tmp3 = np.zeros( ( M, max( N ), V ), dtype = int )
    for m in range( M ):
        tmp3[ m, :N[ m ], : ] =  np.tile( docs[ m ], ( V, 1 ) ).T

    # Define the mask
    mask = ( tmp2 == tmp3 )
 
    # Eventually, mask is a M x max( N ) x V matrix
    # Multiply mask with phi which is a M x max( N ) x k matrix
    # Reshape is required to conduct matrix multiplication

    # Change the size of phi  from M x max( N ) x k to k x M x max( N )
    # Change the size of mask from M x max( N ) x V to V x M x max( N )
    tmp_phi  =  phi.transpose( 2, 0, 1 ).reshape( k, -1 )
    tmp_mask = mask.transpose( 2, 0, 1 ).reshape( V, -1 ).T

    beta = tmp_phi @ tmp_mask

    # Normalization after matrix multiplication to make sum of a given column to 1
    beta_sum = beta.sum( axis = 1, keepdims = True )
    beta = np.divide( beta, beta_sum, out = np.zeros_like( beta ), where=( beta_sum!=0 ) )    

    return alpha, beta


# %% Other Functions

# The variational lower bound
def vlb( docs, phi, gamma, alpha, beta, M, N, k ):
    
    # The L function (Equation 15)
    L_func = 0

    # Over the document
    # Equation 15
    for d in range( M ):

        # Summation of the first line, shown in the L equation of Deepnote
        L_func += gammaln( np.sum( alpha ) ) - np.sum( gammaln( alpha ) ) \
                  + np.sum( [ ( alpha[ i ]    - 1 ) * ( psi( gamma[ d,  i ] ) - psi( np.sum( gamma[ d, : ] ) ) ) for i in range( k ) ] )

        # Summation of the final line, shown in the L equation of Deepnote 
        L_func += -gammaln( np.sum( gamma[ d, : ] ) ) + np.sum( gammaln( gamma[ d, : ] ) ) \
                  - np.sum( [ ( gamma[ d, i ] - 1 ) * ( psi( gamma[ d,  i ] ) - psi( np.sum( gamma[ d, : ] ) ) ) for i in range( k ) ] ) 

        # Summation of the second line, shown in the L equation of Deepnote 
        for n in range( N[ d ] ):

            w_n = int( docs[ d ][ n ] )

            L_func +=  np.sum( [ phi[ d ,n, i ] * ( psi( gamma[ d,  i ] ) - psi( np.sum( gamma[ d, : ] ) ) ) if phi[ d ,n, i ] != 0 else 0 for i in range( k )  ] )
            L_func +=  np.sum( [ phi[ d ,n, i ] * np.log( beta[ i, w_n ] )                                   if beta[ i, w_n ] != 0 else 0 for i in range( k )  ] )
            L_func += -np.sum( [ phi[ d ,n, i ] * np.log( phi[ d, n, i ] )                                   if phi[ d, n, i ] != 0 else 0 for i in range( k )  ] )

    # Taking the average of the documents
    return L_func / M


# %% The Main Loop

def n_most_important( ttmp, vocab, n = 30 ):

    # Get beta with top 9 values
    max_values = ttmp.argsort( )[ -n : ][ ::-1 ]
    return np.array( vocab )[ max_values ]

if __name__ == "__main__":

    # Get argument parser to automize the script
    parser = argparse.ArgumentParser( )
    parser.add_argument( '--run_train' , action='store_true'       )   # To train the LDA model
    parser.add_argument( '--plot'      , action='store_true'       )   # To plot  the LDA model
    parser.add_argument( '--n_train'   , type = int, default = 700 )   # The number of reuters' documents used for the training
    parser.add_argument( '--n_topics'  , type = int, default = 3   )   # The number of topics

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

    # Alpha is the size of the topic, k
    # Note that np.random.rand( k ) can also be used.
    alpha = np.random.gamma( shape = 100, scale = 0.01, size = k ) 

    # Beta is the size of k x V
    # For each topic, it is a dirichlet distribution over verbs
    # The summation should be one, since it is a probability distribution.
    beta  = np.random.dirichlet( np.ones( V ), k )

    # initialize phi and gamma matrices
    # Step (1) of Figure 6
    # Equation 6 (16)
    # phi (The multipomial) is the size of M x max( N ) x k
    # The summation should be one.
    phi = np.ones( ( M, max( N ), k ) ) / k

    # Zero values for other cases
    for m, N_d in enumerate( N ):
        phi[ m, N_d:, : ] = 0 

    # Step (2) of Figure 6
    # Equation 7 (17)
    # Gamma (The Dirichlet Parameters) is the size of M x k
    # N.reshape change N as a coulumn vector
    gamma = alpha + np.ones( ( M,  k ) ) * N.reshape( -1, 1 ) / k

    if is_train: 

        niter = 1000

        # The tolerance to check the L convergence
        tol = 0.1
        L   = -np.inf

        # The list of lb 
        lb_arr = [ ]

        start = time.time()

        for i in range( niter ): 

            # store old value
            L_old = L

            # Conduct the EM Algorithm
            phi  , gamma = E_step( docs, phi, gamma, alpha, beta    )
            alpha, beta  = M_step( docs, phi, gamma, alpha, beta, M )
            
            # check convergence
            L  = vlb( docs, phi, gamma, alpha, beta, M, N, k )
            err = abs( L - L_old )
            
            print( f"Trial:{i: 04}: Variational Lower Bound:{L: .3f}, Delta:{err: .3f}" )
            
            if err < tol:
                break

            lb_arr.append( L )

        end = time.time()

        print( "Training Complete, Ellapsed Time: ", end - start )
        

        # Once the training is complete, save the alpha, beta, gamma, phi, and other variables 
        scipy.io.savemat( "tmp/trained_v" + str( ntrain ) + "_" + str( k ) + "_LDA_EM.mat", { "alpha": alpha, "beta": beta, "gamma":gamma, "phi": phi, "lb_arr" : lb_arr, "M": M, "N": N, "V": V, "ntrain": ntrain } )

    # If no training, then load the trained data and plot the data
    else: 

        # Load the data file
        # Code should be set before hand
        file_name = "dataset/set1/trained_v" + str( ntrain ) + "_LDA_EM.mat"
        data = scipy.io.loadmat( file_name )
        print( "data loaded" )

        # Print out the topic of the d-th document
        for i in range( k ):
            tmp  = data[ "beta" ][ i, : ]
            print( f"TOPIC {i+1:02}: { n_most_important( tmp, vocab, 20 ) }")

        # Choose one document for the print out
        d = 1
        print( ' '.join( reuters.words( id_set[ d ] )  ) )
        print( ' '.join( [ idx_word.get( docs[ d ][ i ] ) for i in range( N[ d ] ) ] ) )

        print( data[ "V" ] )

        # Get the topic from this document.
        # Get the phi matrix of the d-th document, which is max( N ) x k
        tmp_phi = phi[ d, :, :]

        # beta is a k x V matrix
        # Given beta, we can find the topics from the word 
        # Get the best within V words        
        # Plotting the images
        if is_plot:

            # First Figure, plot the beta array, which is k x V
            plt.figure( figsize = (8,8) )
            plt.subplot( 121 )

            # This returns a n_plot_words x k matrix
            sns.heatmap( data[ "beta" ].T, xticklabels = [ ], yticklabels = [ ] )
            plt.xlabel("Topics", fontsize = 14 )
            plt.ylabel( "Words", fontsize = 14 )
            plt.title("topic-word distribution", fontsize = 16 )

            # Second Figure, plot the topic from gamma
            # Gamma is a M x k matrix
            # Given the document, it is a k-length array, which approximates 
            n_sample = 10000

            # Results in n_sample x k matrix, summed over axis 0, which results in a k matrix
            theta_hat = np.array( [ np.random.dirichlet( data[ "gamma" ][ d ], n_sample ).mean( axis = 0 ) for d in range( M ) ] )

            print( data[ "gamma" ].shape  )

            # Normalize over topic
            theta_hat /= theta_hat.sum( axis = 1, keepdims = True )

            plt.subplot(122)
            sns.heatmap( theta_hat, xticklabels=[ ], yticklabels = [ ] )
            plt.xlabel( "Topics"    , fontsize = 14)
            plt.ylabel( "Documents" , fontsize = 14 )
            plt.title( "document-topic distribution", fontsize = 16)
            plt.tight_layout()
            plt.show( )