# A code copy from
# https://github.com/lda-project/lda

import numpy as np
import lda
import lda.datasets

def test( ):
    
    X      = lda.datasets.load_reuters( )
    vocab  = lda.datasets.load_reuters_vocab( )
    titles = lda.datasets.load_reuters_titles( )

    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)

    # model.fit_transform(X) is also available
    model.fit( X )                  

    # model.components_ also works
    topic_word = model.topic_word_  
    
    n_top_words = 8
    
    for i, topic_dist in enumerate( topic_word ):
        topic_words = np.array( vocab )[ np.argsort( topic_dist ) ][ :-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))    


if __name__ == "__main__":
    test( )