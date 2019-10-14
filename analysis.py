#################   DO NOT EDIT THESE IMPORTS #################
import math
import random
import numpy
from collections import *

#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
class HMM:
	"""
	Simple class to represent a Hidden Markov Model.
	"""
	def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
		self.order = order
		self.initial_distribution = initial_distribution
		self.emission_matrix = emission_matrix
		self.transition_matrix = transition_matrix

def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns:
	The file represented as a list of tuples, where each tuple
	is of the form (word, POS-tag).
	A list of unique words found in the file.
	A list of unique POS tags found in the file.
	"""
	file_representation = []
	unique_words = set()
	unique_tags = set()
	f = open(str(filename), "r")
	for line in f:
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		file_representation.append( (word, tag) )
		unique_words.add(word)
		unique_tags.add(tag)
	f.close()
	return file_representation, unique_words, unique_tags


def create_sentences(test_words):
	sentences = []
	begin = 0
	for i in range(len(test_words)):
		if test_words[i] == '.':
			sentence = test_words[begin:i+1]
			begin = i+1
			sentences.append(sentence)
	return sentences

def bigram_viterbi(hmm, sentence):
	"""
	Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
	Inputs:
	  hmm --- the HMM to use to predict the POS of the words in the sentence.
	  sentence ---  a list of words.
	Returns:
	  A list of tuples where each tuple contains a word in the
	  sentence and its predicted corresponding POS.
	"""

	# Initialization
	viterbi = defaultdict(lambda: defaultdict(int))
	backpointer = defaultdict(lambda: defaultdict(int))
	unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
	for tag in unique_tags:
		if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
			viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
		else:
			viterbi[tag][0] = -1 * float('inf')

	# Dynamic programming.
	for t in range(1, len(sentence)):
		backpointer["No_Path"][t] = "No_Path"
		for s in unique_tags:
			max_value = -1 * float('inf')
			max_state = None
			for s_prime in unique_tags:
				val1= viterbi[s_prime][t-1]
				val2 = -1 * float('inf')
				if hmm.transition_matrix[s_prime][s] != 0:
					val2 = math.log(hmm.transition_matrix[s_prime][s])
				curr_value = val1 + val2
				if curr_value > max_value:
					max_value = curr_value
					max_state = s_prime
			val3 = -1 * float('inf')
			if hmm.emission_matrix[s][sentence[t]] != 0:
				val3 = math.log(hmm.emission_matrix[s][sentence[t]])
			viterbi[s][t] = max_value + val3
			if max_state == None:
				backpointer[s][t] = "No_Path"
			else:
				backpointer[s][t] = max_state
	for ut in unique_tags:
		string = ""
		for i in range(0, len(sentence)):
			if (viterbi[ut][i] != float("-inf")):
				string += str(int(viterbi[ut][i])) + "\t"
			else:
				string += str(viterbi[ut][i]) + "\t"

	# Termination
	max_value = -1 * float('inf')
	last_state = None
	final_time = len(sentence) - 1
	for s_prime in unique_tags:
		if viterbi[s_prime][final_time] > max_value:
			max_value = viterbi[s_prime][final_time]
			last_state = s_prime
	if last_state == None:
		last_state = "No_Path"

	# Traceback
	tagged_sentence = []
	tagged_sentence.append((sentence[len(sentence)-1], last_state))
	for i in range(len(sentence)-2, -1, -1):
		next_tag = tagged_sentence[-1][1]
		curr_tag = backpointer[next_tag][i+1]
		tagged_sentence.append((sentence[i], curr_tag))
	tagged_sentence.reverse()
	return tagged_sentence

#####################  STUDENT CODE BELOW THIS LINE  #####################

def read_words_file(filename):
	"""
	splits untagged data into words.
	Input:
	filename --- the file to parse
	Returns:
	The file represented as a list of words.
	"""
	f = open(filename, 'r')
	all_words = map(lambda l: l.split(" "), f.readlines())
	return all_words

def create_sentences(test_words):
	"""
	This function takes a list of words and splits it into lists
	each forming a sentence that ends with a full stop.
	:param test_words:
	:return: a list of list representing sentences
	"""
	sentences = []
	begin = 0
	for i in range(len(test_words)):
		if test_words[i] == '.':
			sentence = test_words[begin:i+1]
			begin = i+1
			sentences.append(sentence)
	return sentences

training,uniqw,uniqt = read_pos_file("training.txt")

def compute_counts(training_data, order):
	"""
	This function takes a list of tuples each representing
	a word and it's tag to train a HMM
	:param training_data: a list of tuples
	:param order: the order of the markov chain
	:return: a tuple of counts, each is a default dictionary of D2 or D3
	"""
	tokens = len(training_data) #number of tokens in training data
	count_tw = defaultdict(lambda: defaultdict(int)) #count of word being taged this way
	count_1t = defaultdict(int) #the number of times a tag appears
	count_2tt = defaultdict(lambda: defaultdict(int))#the number of time two consecutive tags appear
	for token in training_data:
		count_1t[token[1]] += 1
		count_tw[token[1]][token[0]] += 1
	for i in range(len(training_data) - 1):
		count_2tt[training_data[i][1]][training_data[i+1][1]] += 1
	if order == 2:
		return (tokens, count_tw,count_1t, count_2tt)
	elif order == 3:
		count_3ttt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))#the number of times 3 consecutive tags appear
		for i in range(len(training_data) - 2):
			count_3ttt[training_data[i][1]][training_data[i + 1][1]][training_data[i + 2][1]] += 1
		return (tokens, count_tw,count_1t, count_2tt, count_3ttt)

#print compute_counts(training, 2)


def compute_initial_distribution(training_data, order):
	"""
	creates a initial probability distribution matrix based on order
	:param training_data: a list of pairs of word-tags
	:param order: the order of HMM
	:return: a distribution matrix
	"""
	num_sentences = 0
	if order == 2:
		pi_1 = defaultdict(int)
		if len(training_data) > 0: #to make sure there is a word/ sentence
			pi_1[training_data[0][1]] += 1
			num_sentences+= 1
		for i in range(len(training_data) -1):
			if training_data[i][0] == '.':
				pi_1[training_data[i+1][1]] +=1
				num_sentences +=1
		for i in pi_1.keys():
			pi_1[i] /= 1.0*num_sentences
		return pi_1
	elif order == 3:
		pi_2 = defaultdict(lambda : defaultdict(int))
		if len(training_data) > 2: #to make sure there are two words before a dot.
			pi_2[training_data[0][1]][training_data[1][1]] += 1
			num_sentences += 1
		for i in range(len(training_data) -2):
			if training_data[i][0] == '.':
				pi_2[training_data[i+1][1]][training_data[1+2][1]] += 1
				num_sentences += 1
		for i in pi_2.keys():
			for j in pi_2[i].keys():
				pi_2[i][j]/= 1.0*num_sentences
		return pi_2
	'''pi_1 = defaultdict(int)
	if len(training_data) > 0: #to make sure there is a word/ sentence
		pi_1[training_data[0][1]] += 1
	for i in range(len(training_data) -1):
		if training_data[i][0] == '.':
			pi_1[training_data[i+1][1]] +=1
	if order == 2:
		return pi_1
	elif order == 3:
		pi_2 = defaultdict(lambda : defaultdict(int))
		if len(training_data) > 2: #to make sure there are two words before a dot.
			pi_2[training_data[0][1]][training_data[1][1]] += 1
		for i in range(len(training_data) -2):
			if training_data[i][0] == '.':
				pi_2[training_data[i+1][1]][training_data[1+2][1]] += 1
		return pi_2'''

#print compute_initial_distribution([('hey','.')],3)

def compute_emission_probabilities(unique_words, unique_tags, W, C):
	"""
	This function takes lists of all the unique words and unique tags,
	the number of times a word is tagged in some sort, and the number of tags

	:param unique_words: lists of all the unique words in training data
	:param unique_tags: list of all the unique words and unique tags in data
	:param W: the number of times a word is tagged with specific tag
	:param C: the number of times a tag appears.
	:return: the emission probability matrix
	"""
	emission_p_matrix = defaultdict(lambda: defaultdict(int))
	#for tag in unique_tags:
		#for word in unique_words:

	for tag in W.keys():
		for word in W[tag]:
			if C[tag] != 0:
				emission_p_matrix[tag][word] = 1.0*W[tag][word]/C[tag]
	#print "##Wcount##:", W #debuging
	return emission_p_matrix


def return_max_index(lst, order):
	"""
	This function returns the index of the maximum first (order)
	elements in a list
	:param lst: a list of numbers
	:param order: the number of initial elements to check
	:return: an index less than ore equal order (with maximum value)
	"""
	max = float("-INF")
	index = 0
	if order <= len(lst):
		for i in range(order):
			if lst[i] > max:
				index = i
				max = lst[i]
	return index

#print return_max_index([3,4,6,2])

def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
	"""
	This function applies the algorithm compute Lmabda in homework
	description. to be able to smooth values.
	:param unique_tags: a list of unique tags in a set of data.
	:param num_tokens: the number of words tagged
	:param C1: count_1t generated from compute_counts
	:param C2: count_2tt generated from compute_counts
	:param C3: count_3ttt generated from compute_counts
	:param order: the order of markov chain.
	:return: 3 coeefficients for further calculations
	"""
	lambdas = [0, 0, 0]

	#if there are no tokens then lambdas are zero
	if num_tokens == 0:
		return lambdas
	#first order application
	if order == 2:
		for t1 in C2.keys():
			for t2 in C2[t1].keys():
				if C2[t1][t2] != 0:
					alphas = [0,0]
					alphas[0] = 1.0 * (C1[t2] - 1) / num_tokens
					if C1[t1] != 1:  # that is if C1[t1] - 1 != 0
						alphas[1] = 1.0 * (C2[t1][t2] - 1) / (C1[t1] - 1)
					lambdas[return_max_index(alphas, order)] += C2[t1][t2]
	#second order application
	elif order == 3:
	#generate all 3 permutations of tags
		for t0 in C3.keys():
			for t1 in C3[t0].keys():
				for t2 in C3[t0][t1].keys():

					if C3[t0][t1][t2] > 0:
						alphas = [0,0,0]
						alphas[0] = 1.0*(C1[t2]-1)/num_tokens
						if C1[t1] !=  1: #that is if C1[t1] - 1 != 0
							alphas[1] = 1.0*(C2[t1][t2] -1)/(C1[t1] - 1)
						if C2[t0][t1] != 1:
							alphas[2] = 1.0*(C3[t0][t1][t2]-1)/(C2[t0][t1]-1)
						lambdas[return_max_index(alphas, order)] += C3[t0][t1][t2]

	#normalizing lambdas
	sum_lambdas = lambdas[0]+lambdas[1]+lambdas[2]
	for i in range(len(lambdas)):
		lambdas[i] = 1.0*lambdas[i]/ sum_lambdas
	print "lambdas:", lambdas
	return lambdas


############################testing########################
'''training_data1 = read_pos_file("training.txt")
unique = training_data1[1]
tags = training_data1[2]
tuples = training_data1[0]
numtokens = len(tuples)
C1 = compute_counts(training_data1[0], 3)[2]
C2 = compute_counts(training_data1[0], 3)[3]
C3 = compute_counts(training_data1[0], 3)[4]

print compute_lambdas(tags, numtokens , C1, C2, C3, 2)

print compute_lambdas(tags, numtokens , C1, C2, C3, 3)'''

def compute_transition_matrix(unique_tags, num_tokens , C1, C2, C3, order, boolean):
	"""
	This function creates either a first or second transition matrix between tags depending
	on order parameter.
	:param unique_tags:
	:param num_tokens:
	:param C1: count_1t generated from compute_counts
	:param C2: count_2tt generated from compute_counts
	:param C3: count_3ttt generated from compute_counts
	:param order: the order of markov chain.
	:param boolean: whether the function uses smoothing
	:return: a transition Matrix
	"""
	#first order calculations
	if order == 2:
		if boolean == True: #use smoothing
			lambdas = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
		else:
			lambdas = [0,1,0] # no smoothing
		transition_matrix = defaultdict(lambda: defaultdict(int))
		for tag1 in unique_tags:
			for tag2 in unique_tags:
				transition_matrix[tag1][tag2] = (1.0*lambdas[0]*C1[tag2]/num_tokens)
				if C1[tag1]!= 0: #only add if denominator not equal 0
					transition_matrix[tag1][tag2] += (1.0 * lambdas[1] * C2[tag1][tag2] / (C1[tag1]))

	#second order calculations
	elif order == 3:
		if boolean == True:
			lambdas = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
		else:
			lambdas = [0,0,1]
		transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		for tag0 in unique_tags:
			for tag1 in unique_tags:
				for tag2 in unique_tags:
					#check that denominators are not equal to 0
					transition_matrix[tag0][tag1][tag2] = (1.0 * lambdas[0] * C1[tag2] / num_tokens)
					if C2[tag0][tag1] != 0:
						transition_matrix[tag0][tag1][tag2] += (1.0*lambdas[2]*C3[tag0][tag1][tag2]/C2[tag0][tag1])
					if C1[tag1] != 0:
						transition_matrix[tag0][tag1][tag2] += (1.0 * lambdas[1] * C2[tag1][tag2] / (C1[tag1]))
	return transition_matrix




def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	"""
	This function develops a hidden markov model based on training data
	:param training_data: a list of data
	:param unique_tags: a list of tags
	:param unique_words: a list of words
	:param order: the order of the markov chain
	:param use_smoothing: boolean to whther use smoothinf or not
	:return: a HMM trained on data.
	"""
	if order == 2:
		num_tokens, CTW, C1, C2 = compute_counts(training_data, order)
		#print "computeHMMZZZ:", CTW
		C3 = 0 #to eleminate errors
	if order == 3:
		num_tokens, CTW, C1, C2, C3 = compute_counts(training_data, order)
	initial_distribution = compute_initial_distribution(training_data, order)
	emission_matrix = compute_emission_probabilities(unique_words, unique_tags, CTW, C1)
	transition_matrix = compute_transition_matrix(unique_tags, num_tokens , C1, C2, C3, order, use_smoothing)
	model = HMM(order,initial_distribution,emission_matrix,transition_matrix)
	return model

#modelz = build_hmm(training,uniqt,uniqw,2,False)
#print modelz


def trigram_viterbi(hmm, sentence):
	'''
	This function predicts tags of words given a trained data with a second order markov model.
	:param hmm: a hidden markov model
	:param sentence: a test sentence
	:return: the predicted tagging of words in sentences.
	'''
	# Initialization of a 3D Matrix
	viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
	for tag0 in unique_tags:
		for tag1 in unique_tags:
			if (hmm.initial_distribution[tag0][tag1] != 0) and (hmm.emission_matrix[tag0][sentence[0]] != 0) and \
					(hmm.emission_matrix[tag1][sentence[1]]!=0):
				viterbi[tag0][tag1][0] = math.log(hmm.initial_distribution[tag0][tag1]) + math.log(hmm.emission_matrix[tag0][sentence[0]])+ \
										 math.log(hmm.emission_matrix[tag1][sentence[1]])
			else:
				viterbi[tag0][tag1][0] = -1 * float('inf')

	# Dynamic programming.
	for t in range(2, len(sentence)):
		for s2 in unique_tags:
			#backpointer[s2]["No_Path"][t] = "No_Path"
			backpointer["No_Path"][s2][t] = "No_Path"
			for s3 in unique_tags:
				max_value = -1 * float('inf')
				max_state = None
				for s1 in unique_tags:
					val1 = viterbi[s1][s2][t-1]
					val2 = -1 * float('inf')
					if hmm.transition_matrix[s1][s2][s3] != 0:
						val2 = math.log(hmm.transition_matrix[s1][s2][s3])
						curr_value = val1 + val2
						if curr_value > max_value:
							max_value = curr_value
							max_state = s1
					val3 = -1 * float('inf')
					if hmm.emission_matrix[s3][sentence[t]] != 0:
						val3 = math.log(hmm.emission_matrix[s3][sentence[t]])
					viterbi[s2][s3][t] = max_value +val3
					if max_state == None:
						backpointer[s2][s3][t-2] = "No_Path"
					else:
						backpointer[s2][s3][t-2] = max_state

	for ut2 in unique_tags:
		for ut3 in unique_tags:
			string = ""
			for i in range(0, len(sentence)):
				if (viterbi[ut2][ut3][i] != float("-inf")):
					string += str(int(viterbi[ut2][ut3][i])) + "\t"
				else:
					string += str(viterbi[ut2][ut3][i]) + "\t"

	# Termination1, get the arguments with the maximum values for the last two states
	max_value = -1 * float('inf')
	last_state = None
	prelast_state = None
	final_time = len(sentence) - 1
	for s2 in unique_tags:
		for s3 in unique_tags:
			if viterbi[s2][s3][final_time] > max_value:
				max_value = viterbi[s2][s3][final_time]
				last_state = s3
				prelast_state = s2
		if last_state == None:
			last_state = "No_Path"
		if prelast_state == None:
			prelast_state = "No_Path"

	# Traceback1
	#given 2 states, this part assigns the one prior to the that maximizes possibility.
	tagged_sentence = []
	tagged_sentence.append((sentence[len(sentence) - 1], last_state))
	tagged_sentence.append((sentence[len(sentence) - 2], prelast_state))
	for t in range(len(sentence)-3,-1,-1):
		current_state = backpointer[prelast_state][last_state][t]
		tagged_sentence.append((sentence[t],current_state))
		last_state = prelast_state
		prelast_state = current_state

	#to get it in the correct order
	tagged_sentence.reverse()
	return tagged_sentence
###############################################Testing#############################################


def builder_hmm_update(order, percentage, use_smoothing, test_words):
	"""
	This function sets the floor for creating hidden markov model
	and updating it to be prepared for testing phase.
	:param order: the order of the markov model
	:param percentage: the percent of data to be trained
	:param use_smoothing:
	:param test_words:
	:return:
	"""
	tokens,unique_words,unique_tags = read_pos_file("training.txt")
	#test_words = read_words_file("testdata_untagged.txt")
	set_test = set(test_words)
	epsilon = 0.00001
	data = []
	data_set = []
	for i in range(len(tokens)*percentage/100):
		data.append(tokens[i])
		data_set.append(tokens[i][0])
	model = build_hmm(data, unique_tags, unique_words, order, use_smoothing)
	set_data = set(data_set)
	unknowns = set_test.difference(set_data)

	#hmm_update
	for tag in model.emission_matrix.keys():
		for word in unknowns:
			model.emission_matrix[tag][word] = epsilon
	for tag in model.emission_matrix.keys():
		sum_row = 0
		for word in model.emission_matrix[tag].keys():
			if model.emission_matrix[tag][word]!=0:
				model.emission_matrix[tag][word]+= epsilon
				sum_row += model.emission_matrix[tag][word]
		for word in model.emission_matrix[tag].keys():
			model.emission_matrix[tag][word] /= sum_row
	return model

def accuracy_measure(seq1,seq2):
	'''
	to compare accuracy of tested data
	:param seq1: tested
	:param seq2: tagged
	:return: accuracy
	'''
	counter = 0
	length = len(seq1)
	for i in range(length-1):
		if seq1[i][1] == seq2[i][1]:
			counter +=1
	return 1.0*counter/length

def application_order_2(test_file, percentage, use_smoothing):
	'''
	This function calls bigram viterbi and build_hmm
	:param test_file: test data
	:param percentage: percent trained
	:param use_smoothing: whether to smooth data
	:return: bigram
	'''
	test_words = read_words_file(test_file)[0]
	#print test_words
	bigrams = []
	sentences = create_sentences(test_words)
	model = builder_hmm_update(2,percentage,use_smoothing, test_words)
	for i in range(len(sentences)):
		bigram = bigram_viterbi(model,sentences[i])
		bigrams = bigrams + bigram
	print "accuracy", accuracy_measure(bigrams, read_pos_file("testdata_tagged.txt")[0])
	return bigrams


#print application_order_2("testdata_untagged.txt",1,True)

def application_order_3(test_file, percentage, use_smoothing):
	'''
	This function calls trigram_viterbi and build_hmm
	:param test_file: test data
	:param percentage: percent trained
	:param use_smoothing: whether to smooth data
	:return: trigram
	'''
	test_words = read_words_file(test_file)[0]
	#print test_words
	trigrams = []
	sentences = create_sentences(test_words)
	model = builder_hmm_update(3, percentage, use_smoothing, test_words)
	for i in range(len(sentences)):
		trigram = trigram_viterbi(model, sentences[i])
		trigrams = trigrams + trigram
	print "accuracy", accuracy_measure(trigrams, read_pos_file("testdata_tagged.txt")[0])
	return trigrams

def sentence_test(test_file, percentage, use_smoothing):
	'''
	to test trigram viterabi on 1 sentence
	:param test_file: test data
	:param percentage: percent trained
	:param use_smoothing: whether to smooth data
	:return: trigram
	'''
	test_words = read_words_file(test_file)[0]
	#print test_words
	sentences = create_sentences(test_words)
	model = builder_hmm_update(3, percentage, use_smoothing, test_words)
	trigram = trigram_viterbi(model, sentences[0])
	return trigram


#print sentence_test("testdata_untagged.txt",100,False)
#model_test1 =  builder(2,1,False)



#tes_tagged = read_pos_file("testdata_tagged.txt")
#print bigram_viterbi(model_test1, test_untagged)


#############################Experiments#########################
'''All experiments 1 to 4:'''
print "----testing----"
# F_3_100 = application_order_3("testdata_untagged.txt",100,False)
# print "3F100", F_3_100
# T_3_100 = application_order_3("testdata_untagged.txt",100,True)
# print "3T100", T_3_100
# F_3_1 = application_order_3("testdata_untagged.txt",1,False)
# print "3F1", F_3_1
# T_3_1 = application_order_3("testdata_untagged.txt",1,True)
# print "3T1", T_3_1
# F_3_5 = application_order_3("testdata_untagged.txt",5,False)
# print "3F5", F_3_5
# T_3_5 = application_order_3("testdata_untagged.txt",5,True)
# print "3T5", T_3_5
# F_3_10 = application_order_3("testdata_untagged.txt",10,False)
# print "3F10", F_3_10
# T_3_10 = application_order_3("testdata_untagged.txt",10,True)
# print "3T10", T_3_10
# F_3_25 = application_order_3("testdata_untagged.txt",25,False)
# print "3F25", F_3_25
# T_3_25 = application_order_3("testdata_untagged.txt",25,True)
# print "3T25", T_3_25
# F_3_50 = application_order_3("testdata_untagged.txt",50,False)
# print "3F50", F_3_50
# T_3_50 = application_order_3("testdata_untagged.txt",50,True)
# print "3T50", T_3_50
# F_3_75 = application_order_3("testdata_untagged.txt",75,False)
# print "3F75", F_3_75
# T_3_75 = application_order_3("testdata_untagged.txt",75,True)
# print "3T75", T_3_75
# F_2_100 = application_order_2("testdata_untagged.txt",100,False)
# print "2F100", F_2_100
# T_2_100 = application_order_2("testdata_untagged.txt",100,True)
# print "2T100", T_2_100
# F_2_1 = application_order_2("testdata_untagged.txt",1,False)
# print "2F1", F_2_1
# T_2_1 = application_order_2("testdata_untagged.txt",1,True)
# print "2T1", T_2_1
# F_2_5 = application_order_2("testdata_untagged.txt",5,False)
# print "2F5", F_2_5
# T_2_5 = application_order_2("testdata_untagged.txt",5,True)
# print "2T5", T_2_5
# F_2_10 = application_order_2("testdata_untagged.txt",10,False)
# print "2F10", F_2_10
# T_2_10 = application_order_2("testdata_untagged.txt",10,True)
# print "2T10", T_2_10
# F_2_25 = application_order_2("testdata_untagged.txt",25,False)
# print "2F25", F_2_25
# T_2_25 = application_order_2("testdata_untagged.txt",25,True)
# print "2T25", T_2_25
# F_2_50 = application_order_2("testdata_untagged.txt",50,False)
# print "2F50", F_2_50
# T_2_50 = application_order_2("testdata_untagged.txt",50,True)
# print "2T50", T_2_50
# F_2_75 = application_order_2("testdata_untagged.txt",75,False)
# print "2F75", F_2_75
# T_2_75 = application_order_2("testdata_untagged.txt",75,True)
# print "2T75", T_2_75
#print create_sentences(test_words_lst)