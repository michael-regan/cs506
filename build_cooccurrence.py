from collections import deque 
import numpy as np


corpus = [
    'This is the first document',
    'This is the second document',
    'And the third one',
    'Do I strip punctuation?',
]

# corpus = [
#   'My first thought was he lied in every word',
# 	'That hoary cripple with malicious eye',
# 	'Askance to watch the working of his lie'
# ]


all_words=[]
for text in corpus:
	text=text.split()
	for word in text:
		word=word.lower()
		if word not in all_words:
			all_words.append(word)

length=len(all_words)

window = deque(maxlen=2)

m = np.zeros([length,length]) # m x m matrix


def build_cooccurrence(sentence,m):

	sentence = sentence.split()

	for word in sentence:
		word=word.lower()
		
		window.append(word) # this wraps around to next sentence

		thisWindow=list(window)
		if len(thisWindow)>1:
			print(thisWindow)
			# only works for a one-word window
			index_word_0 = all_words.index(thisWindow[0]) 
			index_word_1 = all_words.index(thisWindow[1])

			m[index_word_0][index_word_1]+=1
			m[index_word_1][index_word_0]+=1


for sentence in corpus:
    build_cooccurrence(sentence, m)

np.fill_diagonal(m, 0)

print(m.T[0])
