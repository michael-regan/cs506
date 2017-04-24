from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle, time


neg_path = 'polarityData/rt-polaritydata/negative.txt'

pos_path = 'polarityData/rt-polaritydata/positive.txt'

negDict={}
posDict={}

corpus = []


cnt=0
with open(neg_path, 'r') as f:

	for line in f:
		negDict[cnt]=line
		corpus.append(line)
		cnt+=1


with open(pos_path, 'r') as g:

	for line in g:
		posDict[cnt]=line
		corpus.append(line)
		cnt+=1


# corpus = [
#     'She likes roses .',
#     'She loves tulips .',
#     'We enjoy flowers .',
#     'We enjoy growing roses .'
# ]


def to_csv(myMatrix):

	np.savetxt("/data/mydata.csv", myMatrix)



def build_cooccurrence(sentence,m):

	sentence = sentence.split()

	for word in sentence:
		word=word.lower()
		
		window.append(word) # this wraps around to next sentence

		thisWindow=list(window)
		if len(thisWindow)>2:
			# print(thisWindow)
			# at this point, if you change the window, you have to change this
			index_word_0 = all_words.index(thisWindow[0]) 
			index_word_1 = all_words.index(thisWindow[1])
			index_word_2 = all_words.index(thisWindow[2])

			m[index_word_0][index_word_1]+=1
			m[index_word_1][index_word_0]+=1
			# I don't want to count the following words twice, right?
			# m[index_word_1][index_word_2]+=1
			# m[index_word_2][index_word_1]+=1


def s_v_d(myMatrix):

	t0 = time.time()
	la = np.linalg
	words = all_words
	X = myMatrix
	U, s, V = la.svd(X, full_matrices=False)
	t1 = time.time()

	print("Time for SVD:", t1-t0)

	d25_U=[]
	for i in U:
		d25_U.append(i[:25])
	d25_U=np.array(d25_U)

	pickle.dump(d25_U, open("data/svd_25d", 'wb'), protocol=4)

	print("svd_25d saved to disk")


	d50_U=[]
	for j in U:
		d50_U.append(j[:50])
	d50_U=np.array(d50_U)

	pickle.dump(d50_U, open("data/svd_50d", 'wb'), protocol=4)

	print("svd_50d saved to disk")


	d100_U=[]
	for k in U:
		d100_U.append(k[:100])
	d100_U=np.array(d100_U)

	pickle.dump(d100_U, open("data/svd_100d", 'wb'), protocol=4)

	print("svd_100d saved to disk")


	# print(U)

	# for i in range(len(words)):
	# 	plt.text(U[i,0], U[i,1], words[i])
	# 	print(U[i,0], U[i,1])

	# plt.axis([-0.6, 0, -0.8, 0.7])
	# plt.show()


def plot_histogram(x):

	# buggy

	n, bins, patches = P.hist(x, 50, normed=1, histtype='stepfilled')
	P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

	# add a line showing the expected distribution of tokens with each frequency
	y = P.normpdf( bins, mu, sigma)
	l = P.plot(bins, y, 'k--', linewidth=1.5)

	# plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
	# plt.show()


def to_pickle(words):

	pickle.dump(words, open("data/all_words", 'wb'), protocol=4)



countWords=defaultdict(int)

all_words=[]
for sent in corpus:
	sent=sent.split()
	for word in sent:
		word=word.lower()
		if word not in all_words:
			all_words.append(word)
		# if word not in countWords:
		# 	countWords[word]=1
		# else:
		# 	countWords[word]+=1

# cnt=0
# for k, v in countWords.items():
# 	if cnt<100:
# 		print(k, v)
# 		cnt+=1


# What is the distribution of counts?
# freqOfCounts=defaultdict(int)

# for v in countWords.values():
# 	if v not in freqOfCounts:
# 		freqOfCounts[v]=1
# 	else:
# 		freqOfCounts[v]+=1

# for k, v in freqOfCounts.items():
# 	print(k, v)
# plot_histogram(freqOfCounts)




length=len(all_words)

window = deque(maxlen=3)

m = np.zeros([length,length]) # m x m matrix

for sentence in corpus:
    build_cooccurrence(sentence, m)

np.fill_diagonal(m, 0)

print(m.shape)

# print(all_words)

to_pickle(all_words)

# print(m)

# to_csv(m)

# s_v_d(m)

# print(all_words[0],m.T[0])
# print(all_words[1],m.T[1])
