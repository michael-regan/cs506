import pickle
import numpy as np

# simple experiments with svd word vectors for sentiment analysis
# svd done with svd.py

neg_path = 'polarityData/rt-polaritydata/negative.txt'

pos_path = 'polarityData/rt-polaritydata/positive.txt'

negDict={}
posDict={}


cnt=0
with open(neg_path, 'r') as f:

	for line in f:
		negDict['neg'+str(cnt)]=line
		cnt+=1


with open(pos_path, 'r') as g:

	for line in g:
		posDict['pos'+str(cnt)]=line
		cnt+=1

compiledDict = {**negDict, **posDict}

# length of compiledDict is 10662
# print(len(compiledDict)) 
# print()


data = 'data/dict_100d'

with open(data, 'rb') as h:
	word_with_vectors = pickle.load(h)


compiled_sentence_vectors = {}

errorCnt=0
for k, v in compiledDict.items():
	sentence = v.split()
	length = len(sentence)

	key_error = False

	m = np.zeros(100)

	for word in sentence:

		try:
			word_vec = word_with_vectors[word]
			m+=word_vec

		except KeyError:
			errorCnt+=1
			key_error = True
			break

	# sentence vectors = summing all vectors from sentence and averaging --> the vector mash
	if key_error == False:
		norm_m = m/length
		compiled_sentence_vectors[k] = norm_m


# length: 10499; Key errors: 163
# print(len(compiled_sentence_vectors))
# print("Key errors", errorCnt)

labels = []
vectors = []
vector_ids = []

for k, v in compiled_sentence_vectors.items():

	if 'neg' in k:
		labels.append(0)
	else:
		labels.append(1)

	vectors.append(v)

	vector_ids.append(k)


pickle.dump(labels, open('data/3_labels_100d', 'wb'), protocol=4)
pickle.dump(vectors, open('data/3_vectors_100d', 'wb'), protocol=4)
pickle.dump(vector_ids, open('data/3_vectorIDs_100d', 'wb'), protocol=4)
