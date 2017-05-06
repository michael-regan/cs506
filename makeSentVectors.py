import pickle
import numpy as np
from compileCorpus import compiledDict
from matchWords2Vectors import dict_25d, dict_50d, dict_100d

# simple experiments with svd word vectors for sentiment analysis
# svd done with svd.py


vectors_25d = []
vectors_50d = []
vectors_100d = []

dimensions = [25, 50, 100]

for dimension in dimensions:

	#data = 'data/dict_100d'
	if dimension == 25:
		data = dict_25d
	elif dimension == 50:
		data = dict_50d
	else:
		data = dict_100d

	# with open(data, 'rb') as h:
	# 	word_with_vectors = pickle.load(h)

	word_with_vectors = data
	
	compiled_sentence_vectors = {}

	errorCnt=0
	for k, v in compiledDict.items():
		sentence = v.split()
		length = len(sentence)

		key_error = False

		m = np.zeros(dimension)

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

	# Key errors: 163
	print("Key errors", errorCnt)

	#labels and vectors remain the same for all dimensionalities; vectors will change
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


	if dimension == 25:
		vectors_25d = vectors

	elif dimension == 50:
		vectors_50d = vectors

	else: 
		vectors_100d = vectors


	# pickle.dump(labels, open('data/3_labels_100d', 'wb'), protocol=4)
	# pickle.dump(vectors, open('data/3_vectors_100d', 'wb'), protocol=4)
	# pickle.dump(vector_ids, open('data/3_vectorIDs_100d', 'wb'), protocol=4)
