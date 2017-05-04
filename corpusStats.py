import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from pandas import Series


# preparing basic stats on sentiment analysis corpus

# sentiment data from: https://github.com/shekhargulati/sentiment-analysis-python/tree/master/polarity-data
# small corpus of 21425 unique tokens


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

# dividing into sentences and tokens for distribution (plot)
sentences = []
tokens = []
for k, v in compiledDict.items():
	wordList = v.split()
	sentences.append(wordList)
	for token in wordList:
		tokens.append(token)

allTokens = Series(tokens)
# print(distribution)

counts = allTokens.value_counts()

counts1=0
countsGreater1=0
print("Total tokens:", len(counts))
for k, c in counts.items():
	if c==1:
		counts1+=1
	else:
		countsGreater1+=1
print("# tokens with count 1:", counts1)
print("# tokens with count > 1:", countsGreater1)

#using numpy to divide counts of tokens for histogram
logCounts = np.log(counts)
#print(logCounts)
count, division = np.histogram(logCounts)

plt.figure()
counts.plot.hist(bins=division)
plt.xlabel('log(count)', fontsize=18)
plt.show()
