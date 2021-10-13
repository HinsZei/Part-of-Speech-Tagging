import numpy as np

'''implement viterbi algorithm to predict a possible sequence of POS for sentences'''


def viterbi(test_sents, tags_index, smoothed_transition, smoothed_emission):
    Start = '<s>'
    End = '</s>'
    predict_tags = []  # record the predicted POS , separated by sentences
    '''for each word in a sentence:'''
    for sent in test_sents:
        line_tags = []
        time = 0
        score = {}
        path = {}
        '''
        (time, tag) represent the coordinate of tags in different time step, 
        for each tag, score[time,tag] records the best score to reach it. 
        and the path[time,tag] records the coordinate of the tag connected to 
        it. see below for more detailed explanation
        '''
        score[(0, Start)] = 0
        path[(0, Start)] = None
        '''
        combine the formula:
        v[q; 1] = α(q0,q) * β(q,w1)
        For all t = 2 to n, and for all q: v[q,t] = max v[q',t-1] *α(q', q) * β(q,wt)
        v(qf,n+1) = max v[q',n] *α(q', qf) 
        to prevent underflow, using -log like MLE is a good option. I reserve both versions.
        but actually the original version would underflow even I delete the sentences whose length is over 100.
        
        for initial state time from 0 to n(length of tokens)
        calculate the probability from current_position in T to all the tags in T+1
        then compare the probability to the previous one, and update it. the connection information
        of the best probability is recorded by path[next_position].
        Note that score[(time,tag)] and path[(time,tag)] represent the best probability to reach this tag
        and the last tag connect to this tag
        
        it s ok to use matrix to record the scores and path, but the network contains the only node 
        at the start and the end,which would break the loop and make it too complex, so I didn't implement it.
        '''
        for token in sent:
            for current_tag in tags_index:
                for next_tag in tags_index[1:]:
                    current_position = (time, current_tag)
                    next_position = (time + 1, next_tag)
                    if current_position in score:
                        current_score = score[current_position] + (
                                    -np.log(smoothed_transition[current_tag].prob(next_tag)) - np.log(
                                smoothed_emission[next_tag].prob(token['form'])))
                        # current_score = score[current_position] * smoothed_transition[current_tag].prob(next_tag) * \
                        #                 smoothed_emission[next_tag].prob(token['form'])
                        if next_position not in score or score[next_position] > current_score:
                            score[next_position] = current_score
                            path[next_position] = current_position
            time += 1

        for current_tag in tags_index[1:]:
            current_position = (time, current_tag)
            next_position = (time + 1, End)
            if current_position in score:
                # current_score = score[current_position]*smoothed_transition[current_tag].prob(next_tag)
                current_score = score[current_position] - np.log(smoothed_transition[current_tag].prob(End))
                if next_position not in score or score[next_position] > current_score:
                    score[next_position] = current_score
                    path[next_position] = current_position
        '''once the </s> is visited, we can back-track the path by finding every position starting from </s>'''
        node = path[(time + 1, End)]
        while node != (0, Start):
            tag = node[1]
            line_tags.append(tag)
            node = path[node]
        line_tags.reverse()  # make the sequence consistent with the true POS
        predict_tags.append(line_tags)
    ''' calculate the accuracy'''
    i = 0
    count_correct = 0;
    prediction = sum(predict_tags, [])  # flat the 2-dimensional list to 1-dimensional list
    for sent in test_sents:
        for token in sent:
            if prediction[i] == token['upos']:
                count_correct += 1
            i += 1
    accuracy = count_correct / i
    print('the accuracy is: ' + str(accuracy))
