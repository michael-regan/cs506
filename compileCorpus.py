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
