import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk import WittenBellProbDist

'''
HMM model consists of 3 parts:α and β and the initial probability Π(s),which is contained in the transition matrix, 
represented as a row of <s> .
The method aims to these parameters.
'''
def train_hmm(train_sents):
    tags = []
    words = []
    emission = []
    transition = []
    # define the start and the end of a single sentence
    Start = '<s>'
    End = '</s>'
    for sent in train_sents:
        # tags.append(Start)
        for token in sent:
            tags.append(token['upos']);
            words.append(token['form'])
    # tags.append(End)
    '''
    tags_col: index of the transition matrix,as there is no tag to follow the '<s>', only the '</s>' is added
    tags_row: index of the transition matrix,as there is no tag to follow the '</s>', only the '<s>' is added
    words_index:index of the emission matrix
    count transiton/emission: the matrix of the count of tags and words, 
                              which is used to show data instead of calculating the probability matrix
    '''
    tags_col = list(FreqDist(tags).keys())
    tags_row = tags_col[:]
    tags_col.append(End)
    tags_row.insert(0, Start)
    words_index = list(FreqDist(words).keys())
    count_transition = np.zeros((len(tags_row), len(tags_col)))
    count_emission = np.zeros((len(tags_row), len(words_index)))
    '''
    transition: [(previous_tag,tag)]. combine the previous tag with tag for generating the probability matrix.
    <s> and </s> are included.
    emission: [(tag,word)].
    at the begin of a sentence,set previous_tag = <s> to ensure transition contain initial probability 
    add </s> manually at the end.
    '''
    for sent in train_sents:
        previous_tag = Start
        for token in sent:
            tag, word = token['upos'], token['form']
            transition.append((previous_tag, tag))
            emission.append((tag, word))
            count_transition[tags_row.index(previous_tag)][tags_col.index(tag)] += 1
            # count_emission[tags_row.index(tag)][words_index.index(word)] += 1
            previous_tag = tag
        transition.append((previous_tag, End))
        #
        count_transition[tags_row.index(previous_tag)][tags_col.index(End)] += 1
    df_transition = pd.DataFrame(count_transition, columns=tags_col, index=tags_row)
    # df_emission = pd.DataFrame(count_emission, columns=words_index, index=tags_row)
    # df_emission.drop(Start, axis=0, inplace=True)
    print(df_transition)
    # print(df_emission)
    '''abandoned because can't keep the words and tags the same length, and dont know where to add <s> </s>'''
    # for i in range(1,len(tags)):
    # 	tag = tags[i]
    # 	word = words[i]
    # 	previous_tag = tags[i-1]
    # 	if previous_tag != End:
    # 		transition.append((previous_tag,tag))
    # 	if tag != Start and tag != End:
    # 		emission.append((tag,word))
    # 	count_transition[tags_index.index(previous_tag)][tags_index.index(tag)] +=1
    # 	count_emission[tags_index.index(tag)][words_index.index(word)] +=1
    # df_transition = pd.DataFrame(count_transition, columns=tags_index, index=tags_index)
    # df_transition.drop(Start,inplace=True,axis=1)
    # df_transition.drop(End,inplace=True)
    # df_emission = pd.DataFrame(count_emission, columns=words_index, index=tags_index)
    '''
    using WittenBellProbDist to get a probability matrix(actually not a matrix)
    to get the value, use   smoothed_emission[tag].prob(word/tag)
    '''
    smoothed_transition = {}
    smoothed_emission = {}
    for tag in tags_row:
        taged_words = [w for (t, w) in emission if t == tag]
        taged_tag = [t for (pt, t) in transition if pt == tag]
        smoothed_emission[tag] = WittenBellProbDist(FreqDist(taged_words), bins=1e5)
        smoothed_transition[tag] = WittenBellProbDist(FreqDist(taged_tag), bins=1e5)
    # print(smoothed_emission['PROPN'].prob('Google'))
    return smoothed_transition, smoothed_emission, tags_row
