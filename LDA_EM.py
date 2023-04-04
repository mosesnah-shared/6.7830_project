# %% Import the necessary libraries
import os
import nltk
import time

import warnings

import numpy             as np
import matplotlib.pyplot as plt
import seaborn 			 as sns

# Set the random seed    
np.random.seed( 0 )

# Download the nltk corpus
nltk.download( 'reuters'   )
nltk.download( 'stopwords' )

# "After" download, import the reuters and stopwords
# The details are contained in:
# [REF] https://www.nltk.org/book/ch02.html
from nltk.corpus import reuters
from nltk.corpus import stopwords

from scipy.special import psi, polygamma, gammaln

# %% Define stopwords not included as a vocabularies

stops = stopwords.words("english")
stops += [
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
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
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", ".", "!", "?", ",", ";", ":", "[", "]", "{", "}", "-", "+", 
    "_", "/", "@", "#", "$", "%", "^", "&", "*", "(", ")", "<", ">", "|", "=",
    ".-", ".,", "'", '"', ',"'
]

# %% Read the training set and test set

# Distinguish the training/test set of the Reuter's data set
trainset, testset = [ ], [ ]

# The vocabulary list, produced from the training set 
vocab = [ ]

# Hyperparameter: the number of training set of the document
ntrain = 2000

# Counter
i = 0

for file_id in reuters.fileids( ):

    if file_id.startswith( "train" ):

        # Save the words, except the (1) stopwords (2) numbers
        doc = [ w.lower( ) for w in reuters.words( file_id ) if ( w.isupper( ) ) and ( w.lower( ) not in stops ) and ( not w.isnumeric( ) ) ]

        # Save the values
        if doc:
            trainset.append( doc )
            vocab += doc
            i += 1

    else:
        testset.append( [ w.lower( ) for w in reuters.words( file_id ) if ( w.isupper( ) ) and ( w.lower( ) not in stops ) and ( not w.isnumeric( ) ) ] )

    if i >= ntrain:
        break

# Within the vocabulary list, get only the unique vocabularies.
vocab = list( set( vocab ) )

# The number of vocabularies in the dictionary
# As with the LDA paper, we use "V" for the number of "words" (vocabs)
V = len( vocab )

# Produce a dictionary, with key-value pair as word-index (word_idx) pair.
word_idx = { w : i for i, w in enumerate( vocab ) }

# Change the trainset/testset documents from a list of words to a list of index
data = dict( )
data[ "train" ] = [ np.array( [ word_idx.get( w, V ) for w in doc ] ) for doc in trainset ] 
data[ "test"  ] = [ np.array( [ word_idx.get( w, V ) for w in doc ] ) for doc in  testset ] 

# To train the LDA algorithm we use the "train" set
docs = data[ "train" ]

# The numpy array of the length of each document
N = np.array( [ len( doc ) for doc in docs ] )

# The number of documents for the train set. 
# Note that this should be identical to ntrain
M = len( N )

# number of topics
k = 10  

# %% Initialize the alpha, beta, phi, gamma
# Alpha is the size of the topic, k
# Note that np.random.rand( k ) can also be used.
alpha = np.random.gamma( shape = 100, scale = 0.01, size = k ) 

# Beta is the size of k x V
# For each topic, it is a dirichlet distribution over verbs
# The summation should be one, since it is a probability distribution.
beta  = np.random.dirichlet( np.ones( V ), k )

# initialize ϕ, γ
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


# %% Define the E-step of the EM algorithm
def E_step( docs, phi, gamma, alpha, beta ):

    # Optimize phi
    # The number of topics:
    k = len( alpha )
    
    for m in range( M ):

        # "phi" is a 2D array
        # Iterating over the words of the m-th document
        for n in range( N[ m ] ):

            # Iterating over the topics
            for i in range( k ):
                phi[ m, n, i ] = beta[ i, docs[ m ][ n ] ] * np.exp( psi( gamma[ m, i ] ) - psi( gamma[ m, : ].sum( ) ) )

            # Once the calculation is complete, normalize over the 3rd axis
            # phi[ m, n, : ] /= phi[ m, n, : ].sum( )

    if np.any( np.isnan( phi ) ):
        raise ValueError( "phi nan" )

    # Optimize gamma
    # Equation 17, sum over the document
    for m in range( M ):
        for i in range( k ):
            gamma[ m, i ] = alpha[ i ] + phi[ m, :, i ].sum( )

    # ========================================================

    return phi, gamma

def E_step_opt( docs, phi, gamma, alpha, beta ):

    # Phi is a M x max( N ) x k array
    # Beta is a k x V array
    # Gamma is a M x k array
    # But to simplify the calculation, 

    # Create a 3D beta array, to a size of M x max( N ) x k
    beta3D = np.zeros( ( M, max( N ), k ) )

    for i in range( M ):
        beta3D[ i, :N[ i ], : ] = beta[ :, docs[ i ] ].T

    # Create Gamma, also to a size of M x max( N ) x k
    # First, calculate the 2D array 
    tmp = np.exp( psi( gamma ) - psi( gamma.sum( axis = 1 ) )[ :, None ] )

    # Expand this array to axis 1 with max( N )
    gamma3D = np.stack( [ tmp for _ in range( max( N ) ) ], axis = 1 )

    # Update phi, based on the equation
    phi = beta3D * gamma3D

    # Normalize along the topic
    # phi /= sum( phi, axis = 2 )
    
    # Update Gamma
    # Gamma is a M x k matrix
    # Alpha is a k array 
    # Phi is a M x max( N ) x k array

    # Replicate the alpha as a M x k array
    # Summation over axis = 1
    gamma = np.tile( alpha, ( M, 1 ) ) + phi.sum( axis = 1 )

    return phi, gamma


# %% Define the M-step of the EM algorithm

def M_step( docs, phi, gamma, alpha, beta, M ):
    """
        maximize the lower bound of the likelihood.
        This is the M-step of variational EM algorithm for (smoothed) LDA.    
        update of alpha follows from appendix A.2 of Blei et al., 2003.
    """

    # =================================
    # update alpha
    # =================================
    # Set maximum iterations as 10000
    max_iter = 10000
    tol = 1e-6

    # The number of topics
    k = len( alpha )

    for _ in range( max_iter ):

        # store old value
        alpha_old = alpha.copy( )

        # Calculate h array
        h = -M * polygamma( 1, alpha )

        # Calculate gradient (g) array
        g = M * psi( alpha.sum( ) ) - M * psi( alpha ) + ( psi( gamma ) - psi( gamma.sum( axis = 1 ) ) ).sum( axis = 0 )

        # Calculate c 
        c = sum( g / h ) / ( 1.0/z + sum( 1.0/h ) )

        alpha -= ( g - c ) / h

        # check convergence
        err = np.sqrt( np.mean( ( alpha - alpha_old ) ** 2 ) )

        # Check if error smaller than tolerance
        crit = err < tol
        if crit:
            break

    else:
        warnings.warn(f"max_iter={max_iter} reached: values might not be optimal.")
    
    # =================================
    # update beta
    # =================================

    # Define phi_{dni}w_{dn}^{j} first
    # phi[ :, :, i ]

    # Iterating over the vocabularies
    for j in range( V ):
        beta[ :, j ] = np.array( [ phi_dot_w( docs, phi, m, j ) for m in range( M ) ] ).sum( axis = 0 )

    beta /= beta.sum(axis=1).reshape(-1, 1)


    return alpha, beta

def M_step_ref( docs, phi, gamma, alpha, beta, M ):

    # update alpha
    alpha = update( alpha, gamma, M )
    
    # update beta
    for j in range( V ):
        beta[ :, j ] = np.array( [ phi_dot_w(docs, phi, m, j) for m in range(M)]).sum(axis=0)
    beta /= beta.sum(axis=1).reshape(-1, 1)

    return alpha, beta

def update(var, vi_var, const, max_iter=10000, tol=1e-6):
    """
    From appendix A.2 of Blei et al., 2003.
    For hessian with shape `H = diag(h) + 1z1'`
    
    To update alpha, input var=alpha and vi_var=gamma, const=M.
    To update eta, input var=eta and vi_var=lambda, const=k.
    """
    for _ in range(max_iter):
        # store old value
        var0 = var.copy()
        
        # g: gradient 
        psi_sum = psi(vi_var.sum(axis=1)).reshape(-1, 1)
        g = const * (psi(var.sum()) - psi(var)) \
            + (psi(vi_var) - psi_sum).sum(axis=0)

        # H = diag(h) + 1z1'
        z =  const * polygamma(1, var.sum())  # z: Hessian constant component
        h = -const * polygamma(1, var)        # h: Hessian diagonal component
        c = (g / h).sum() / (1./z + (1./h).sum())

        # update var
        var -= (g - c) / h
        
        # check convergence
        err = np.sqrt(np.mean((var - var0) ** 2))
        crit = err < tol
        if crit:
            break
    else:
        warnings.warn(f"max_iter={max_iter} reached: values might not be optimal.")
    
    #print(err)
    return var

def phi_dot_w( docs, phi, d, j ):
    """
        \sum_{n=1}^{N_d} ϕ_{dni} w_{dn}^j
    """
    return ( docs[ d ] == j ) @ phi[ d, :N[ d ], :]


# %% Other Functions
def dg( gamma, d, i ):
    """
        E[log θ_t] where θ_t ~ Dir(gamma)
    """
    # Equation in Section A.1.
    return psi( gamma[ d,  i ] ) - psi( np.sum( gamma[ d, : ] ) )


def dl(lam, i, w_n):
    """
        E[log β_t] where β_t ~ Dir( lam )
    """
    return psi( lam[ i, w_n ] ) - psi( np.sum( lam[ i, : ] ) )

def vlb(docs, phi, gamma, alpha, beta, M, N, k):
    """
    Average variational lower bound for joint log likelihood.
    """
    lb = 0

    # Over the document
    # Equation 15
    for d in range( M ):
        lb += (
            gammaln( np.sum( alpha ) ) - np.sum( gammaln( alpha ) ) + np.sum( [ ( alpha[ i ] - 1) * dg(gamma, d, i) for i in range(k)])
        )

        lb -= (
            gammaln(np.sum(gamma[d, :]))
            - np.sum(gammaln(gamma[d, :]))
            + np.sum([(gamma[d, i] - 1) * dg(gamma, d, i) for i in range(k)])
        )

        for n in range( N[ d ] ):
            w_n = int( docs[ d ][ n ] )

            lb += np.sum( [phi[d][n, i] * dg(gamma, d, i) for i in range(k)])
            lb += np.sum( [phi[d][n, i] * np.log(beta[i, w_n]) for i in range(k)])
            lb -= np.sum( [phi[d][n, i] * np.log(phi[d][n, i]) for i in range(k)])

    return lb / M

# %% [markdown]
# #### Training

# %% [markdown]
# * Only on 2,000 documents

# %%

# for plots later, reorder training set
# (don't need to do this)

# if "lda_trainset.idx" in os.listdir():
#     with open("lda_trainset.idx") as r:
#         idx = eval(r.read())

#     docs = np.array( data[ "train" ] )[ idx ].tolist( )
# else:
    

# %%

N_EPOCH = 1000
TOL = 0.1

verbose = True
lb = -np.inf

phi1 = phi.copy( )
phi2 = phi.copy( )

gamma1 = gamma.copy( )
gamma2 = gamma.copy( )

alpha1 = alpha.copy( )
alpha2 = alpha.copy( )

beta1 = beta.copy( )
beta2 = beta.copy( )

for epoch in range( N_EPOCH ): 

    # store old value
    lb_old = lb

    # The E-step
    phi1, gamma1 =      E_step( docs, phi1, gamma1, alpha1, beta1 )
    phi2, gamma2 =  E_step_opt( docs, phi2, gamma2, alpha2, beta2 )
  
    # Compare 
    for i in range( phi1.shape[ 0 ] ):
        for j in range( phi1.shape[ 1 ] ):
            for l in range( phi1.shape[ 2 ] ):
                assert( phi1[ i, j, l ] == phi2[ i, j, l ] )
    
    print( "Match! For epoch: " + str( epoch ) )

    # The M-step
    # alpha1, beta1  =     M_step( docs, phi, gamma, alpha1, beta1, M )    
    # alpha2, beta2  = M_step_ref( docs, phi, gamma, alpha2, beta2, M )

    # assert( all( abs( alpha1 - alpha2 ) <= 1e-9 ) ) 
    # print( "alpha match!" )


    # check anomaly
    if np.any( np.isnan( alpha1 ) ):
        print( "NaN detected: alpha" )
        break
    
    # check convergence
    lb  = vlb( docs, phi, gamma, alpha1, beta1, M, N, k )
    err = abs( lb - lb_old )
    
    # check anomaly
    if np.isnan( lb ):
        print("NaN detected: lb")
        break
        
    if verbose:
        print(f"{epoch: 04}:  : {lb: .3f},  error: {err: .3f}")
    
    if err < TOL:
        break

else:
    warnings.warn(f"max_iter reached: values might not be optimal.")

print(" ========== TRAINING FINISHED ==========")

# %% [markdown]
# * Training result

# %% [markdown]
# 1. Topic extraction

# %%
def n_most_important(beta_i, n=30):
    """
    find the index of the largest `n` values in a list
    """
    
    max_values = beta_i.argsort()[-n:][::-1]
    return np.array(vocab)[max_values]

# %%
for i in range(k):
    print(f"TOPIC {i:02}: {n_most_important(beta[i], 9)}")

# %% [markdown]
# 2. Topic-word & document-document distribution

# %%
n_sample = 10000
theta_hat = np.array([np.random.dirichlet(gamma[d], n_sample).mean(0) for d in range(M)])
theta_hat /= theta_hat.sum(1).reshape(-1, 1)

# %%
plt.figure(figsize=(8,8))
plt.subplot(121)
n_plot_words = 150
sns.heatmap(beta.T[:n_plot_words], xticklabels=[], yticklabels=[])
plt.xlabel("Topics", fontsize=14)
plt.ylabel(f"Words[:{n_plot_words}]", fontsize=14)
plt.title("topic-word distribution", fontsize=16)

plt.subplot(122)
sns.heatmap(theta_hat, xticklabels=[], yticklabels=[])
plt.xlabel("Topics", fontsize=14)
plt.ylabel("Documents", fontsize=14)
plt.title("document-topic distribution", fontsize=16)

plt.tight_layout();
# %%