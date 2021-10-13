from io import open
from conllu import parse_incr
from Hmm import train_hmm
from Viterbi import viterbi
import sys


corpora = {}
corpora['en'] = 'UD_English-EWT/en_ewt'
corpora['es'] = 'UD_Spanish-GSD/es_gsd'
corpora['nl'] = 'UD_Dutch-Alpino/nl_alpino'
corpora['cantonese'] = 'UD_Cantonese-HK/yue_hk'


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
max_len = 30
if lang == 'cantonese':
    train_sents = conllu_corpus(test_corpus(lang))[0:700]
    test_sents = conllu_corpus(test_corpus(lang))[701:]
else:
    train_sents = conllu_corpus(train_corpus(lang))
    test_sents = conllu_corpus(test_corpus(lang))
test_sents = [sent for sent in test_sents if len(sent) <= max_len]
smoothed_transition, smoothed_emission, tags_index = train_hmm(train_sents)
viterbi(test_sents, tags_index, smoothed_transition, smoothed_emission)
