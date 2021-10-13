import numpy as np
import pandas as pd
from nltk import FreqDist
from io import open
from conllu import parse_incr
import sys
from matplotlib import pyplot as plt
corpora = {}
corpora['en'] = 'UD_English-EWT/en_ewt'
corpora['es'] = 'UD_Spanish-GSD/es_gsd'
corpora['nl'] = 'UD_Dutch-Alpino/nl_alpino'
corpora['can'] = 'UD_Cantonese-HK/yue_hk'
corpora['ch_hk'] = 'UD_Chinese-HK/zh_hk'
corpora['greek'] = 'UD_Greek-GDT/el_gdt'
def train_corpus(lang):
    return corpora[lang] + '-ud-train.conllu'


def test_corpus(lang):
    return corpora[lang] + '-ud-test.conllu'


# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]

if len(sys.argv) == 2 :
    lang = sys.argv[1]
else:
    # Choose language.
    lang = 'en'

# Limit length of sentences to avoid underflow.
max_len = 50
train_sents = conllu_corpus(test_corpus(lang))


lemma = []
count = 0
for sent in train_sents:
    # tags.append(Start)
    for token in sent:
        lemma.append(token['lemma'])
        count +=1
lemma_cant = list(set(lemma))
length1 = len(lemma_cant)/count
lang = 'greek'
lemma = []
train_sents = conllu_corpus(test_corpus(lang))
count = 0
for sent in train_sents:
    # tags.append(Start)
    for token in sent:
        lemma.append(token['form'])
        count+=1
lemma_ch_hk = list(set(lemma))
length2 = len(lemma_ch_hk)/count
x = ('English','Greek')
y = (length1,length2)

plt.bar(x,y)

for a,b in zip(x,y):
    plt.text(a,b,'%.3f' % b)
plt.savefig('en.jpg')
plt.show()

# tags = []
# words = []
# # define the start and the end of a single sentence
# Start = '<s>'
# End = '</s>'
# for sent in train_sents:
#     # tags.append(Start)
#     for token in sent:
#         tags.append(token['upos']);
#         words.append(token['form'])


# tags_col = list(FreqDist(tags).keys())
# tags_row = tags_col[:]
# tags_col.append(End)
# tags_row.insert(0, Start)
# words_index = list(FreqDist(words).keys())
# count_transition = np.zeros((len(tags_row), len(tags_col)))
# count_emission = np.zeros((len(tags_row), len(words_index)))
# '''
# transition: [(previous_tag,tag)]. combine the previous tag with tag for generating the probability matrix.
# <s> and </s> are included.
# emission: [(tag,word)].
# at the begin of a sentence,set previous_tag = <s> to ensure transition contain initial probability
# add </s> manually at the end.
# '''
# for sent in train_sents:
#     previous_tag = Start
#     for token in sent:
#         tag, word = token['upos'], token['form']
#         count_transition[tags_row.index(previous_tag)][tags_col.index(tag)] += 1
#         # count_emission[tags_row.index(tag)][words_index.index(word)] += 1
#         previous_tag = tag
#     count_transition[tags_row.index(previous_tag)][tags_col.index(End)] += 1
# df_transition = pd.DataFrame(count_transition, columns=tags_col, index=tags_row)
# # df_emission = pd.DataFrame(count_emission, columns=words_index, index=tags_row)
# # df_emission.drop(Start, axis=0, inplace=True)
# print(df_transition)
tags = []
words = []
# define the start and the end of a single sentence
Start = '<s>'
End = '</s>'
count = 0
for sent in train_sents:
    # tags.append(Start)
    for token in sent:
        tags.append(token['upos']);
        words.append(token['form'])
        count += 1


tags_col = list(FreqDist(tags).keys())
tags_row = tags_col[:]
tags_col.append(End)
tags_row.insert(0, Start)
words_index = list(FreqDist(words).keys())
count_transition = np.zeros((len(tags_row), len(tags_col)))
count_emission = np.zeros((len(tags_row), len(words_index)))

def color_range(val):
    if val > 0.05:
        color = 'orange'
    else:
      color = 'white'
    return 'background-color: %s' % color


for sent in train_sents:
    previous_tag = Start
    for token in sent:
        tag, word = token['upos'], token['form']
        count_transition[tags_row.index(previous_tag)][tags_col.index(tag)] += 1
        # count_emission[tags_row.index(tag)][words_index.index(word)] += 1
        previous_tag = tag
    count_transition[tags_row.index(previous_tag)][tags_col.index(End)] += 1
df_transition = pd.DataFrame(count_transition, columns=tags_col, index=tags_row)
# df_emission = pd.DataFrame(count_emission, columns=words_index, index=tags_row)
df_transition = df_transition/count
# df_emission.drop(Start, axis=0, inplace=True)
# df_transition.style.applymap(color_range)