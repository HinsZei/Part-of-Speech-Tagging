# Part of Speech Tagging
 A tool for generating POS of any text based on the UD treebanks.Refurbished version of CS5012 P1 in the University of St Andrews

## Intro
 The tool  developed a **first-order HMM** (Hidden Markov Model) for POS (part of speech) tagging in Python using Universal Dependencies Treebanks.

 Parameters in the HMM model includes initial probability Π, transition probability α, emission probability β. The probability of the initial state, which can be derived by counting the initial tag of each sentence in the corpus in supervised learning, and can be merged into the transition matrix. The transition matrix represents the probability of changing from one hidden state to another, while the emission matrix is the probability that the current state corresponds to a particular observation.

 This project generates the set of tuples consisting of a label and the next label, as well as the set of labels and tuples of corresponding words, and then gets the transition and emission through WittenBellProbDist().

 Once the HMM model has trained, it can be solved using **Viterbi**, which treats the probability of moving from one state to another as the length of a path, and finds the longest path, the node of which is the prediction. In particular, the probability is changed to the Log function to prevent underflow, so that the algorithm subsequently calculates the shortest path.
