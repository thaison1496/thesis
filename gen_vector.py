import numpy as np
import pickle
from alphabet import Alphabet
import codecs

def map_number_and_punct(word):
	if any(char.isdigit() for char in word):
		word = u'<number>'
	elif word in [u',', u'<', u'.', u'>', u'/', u'?', u'..', u'...', u'....', u':', u';', u'"', u"'", u'[', u'{', u']',
				  u'}', u'|', u'\\', u'`', u'~', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'-', u'+',
				  u'=']:
		word = u'<punct>'
	return word


# output an list of sentences, each sentence is a list of element (word, tag, pos, ...)
# for example: [['Cau', 'hoi', thu', 'nhat','.'], ['Cau', 'hoi', 'thu', 'hai','.'], ...]
def read_conll_format(input_files):
	word_list = []
	chunk_list = []
	pos_list = []
	tag_list = []
	num_sent = 0
	max_length = 0
	for input_file in input_files:
		with codecs.open(input_file, 'r', 'utf-8') as f:
			words = []
			chunks = []
			poss = []
			tags = []
			for line in f:
				line = line.strip().split("\t")
				if len(line) > 1:
					# change all puctions and numbers to <punct> and <num>
					words.append(map_number_and_punct(line[0].lower()))
					poss.append(line[1])
					chunks.append(line[2])
					# ignore MIC
					if 'MIC' in line[3]:
						line[3] = 'O'
					tags.append(line[3])
				else:
					word_list.append(words)
					pos_list.append(poss)
					chunk_list.append(chunks)
					tag_list.append(tags)
					sent_length = len(words)
					words = []
					chunks = []
					poss = []
					tags = []
					num_sent += 1
					max_length = max(max_length, sent_length)
	return word_list, pos_list, chunk_list, tag_list, num_sent, max_length


# map each value to an unique id, for example: Np -> 1, N -> 2, ...
# and update alphabet
def map_string_2_id(string_list, alphabet):
	string_id_list = []
	for strings in string_list:
		ids = []
		for string in strings:
			id = alphabet.get_index(string)
			ids.append(id)
		string_id_list.append(ids)
	return string_id_list


# from word to embedding
def construct_tensor_word(word_sentences):
	X = np.empty([len(word_sentences), max_length, embedd_dim])
	for i in range(len(word_sentences)):
		words = word_sentences[i]
		length = len(words)
		for j in range(length):
			word = words[j].lower()
			try:
				embedd = embedd_vectors[embedd_words.index(word)]
			except:
				embedd = unknown_embedd
			X[i, j, :] = embedd
		# Zero out X after the end of the sequence
		X[i, length:] = np.zeros([1, embedd_dim])
	return X

# from word to embedding index
def word_to_index(word_sentences):
	X = np.full((len(word_sentences), max_length), zero_embedding_pos)
	for i in range(len(word_sentences)):
		words = word_sentences[i]
		length = len(words)
		for j in range(length):
			word = words[j].lower()
			try:
				embedd_index = embedd_words.index(word)
			except:
				embedd_index = unknown_embedd_pos
			X[i, j] = embedd_index
	return X

# from str to onehot vector (for pos, chunk, tag)
def construct_tensor_onehot(feature_sentences, alphabet, dim):
	X = np.zeros([len(feature_sentences), max_length, dim])
	for i in range(len(feature_sentences)):
		for j in range(len(feature_sentences[i])):
			idx = alphabet.get_index(feature_sentences[i][j])
			if idx > 0:
				X[i, j, idx] = 1
	return X


def load_embedding():
	global embedd_words
	global embedd_vectors
	global unknown_embedd
	global embedd_dim
	word_dir = '../vie-ner-lstm/embedding/words.pl'
	vector_dir = '../vie-ner-lstm/embedding/vectors.npy'
	embedd_vectors = np.load(vector_dir)
	with open(word_dir, 'rb') as handle:
		embedd_words = pickle.load(handle)
	embedd_dim = np.shape(embedd_vectors)[1]
	unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])

	# print(embedd_words[:10])
	# print(embedd_vectors[:10])

def get_domain(files):
	file_names = []
	file_domains = []
	for file in files:
		if type(file) is str:
			file_names.append(file)
			file_domains.append(1)
		else:
			file_names.append(file[0])
			file_domains.append(file[1])
			

def load_data(train_files, dev_files, test_files):
	global train_domain, dev_domain, test_domain
	global train_word, train_pos, train_chunk, train_tag, train_num_sent, train_max_length
	global dev_word, dev_pos, dev_chunk, dev_tag, dev_num_sent, dev_max_length
	global test_word, test_pos, test_chunk, test_tag, test_num_sent, test_max_length
	global max_length

	train_files, train_domain = get_domain(train_files)
	dev_files, dev_domain = get_domain(dev_files)
	test_files, test_domain = get_domain(test_files)

	train_word, train_pos, train_chunk, train_tag, train_num_sent, train_max_length = \
	read_conll_format(train_files)

	dev_word, dev_pos, dev_chunk, dev_tag, dev_num_sent, dev_max_length = \
	read_conll_format(train_files)\

	test_word, test_pos, test_chunk, test_tag, test_num_sent, test_max_length = \
	read_conll_format(test_files)

	max_length = max(train_max_length, test_max_length, dev_max_length)


# create pos, chunk, tags str to id mapping for whole data
def str_to_id():
	global alphabet_pos, alphabet_chunk, alphabet_tag

	alphabet_pos = Alphabet('pos')
	train_pos_id = map_string_2_id(train_pos, alphabet_pos)
	alphabet_pos.close()
	dev_pos_id = map_string_2_id(dev_pos, alphabet_pos)
	test_pos_id = map_string_2_id(test_pos, alphabet_pos)

	alphabet_chunk = Alphabet('chunk')
	train_chunk_id = map_string_2_id(train_chunk, alphabet_chunk)
	alphabet_chunk.close()
	dev_chunk_id = map_string_2_id(dev_chunk, alphabet_chunk)
	test_chunk_id = map_string_2_id(test_chunk, alphabet_chunk)

	alphabet_tag = Alphabet('tag')
	train_tag_id = map_string_2_id(train_tag, alphabet_tag)
	alphabet_tag.close()
	dev_tag_id = map_string_2_id(dev_tag, alphabet_tag)
	test_tag_id = map_string_2_id(test_tag, alphabet_tag)


# def str_to_vec():
# 	global input_train, output_train, input_dev, output_dev, input_test, output_test
# 	train_word_v = construct_tensor_word(train_word)
# 	dev_word_v = construct_tensor_word(dev_word)
# 	test_word_v = construct_tensor_word(test_word)

# 	dim_pos = alphabet_pos.size()
# 	dim_chunk = alphabet_chunk.size()
# 	dim_tag = alphabet_tag.size()

# 	train_pos_v = construct_tensor_onehot(train_pos, alphabet_pos, dim_pos)
# 	train_chunk_v = construct_tensor_onehot(train_chunk, alphabet_chunk, dim_chunk)
# 	train_tag_v = construct_tensor_onehot(train_tag, alphabet_tag, dim_tag)

# 	dev_pos_v = construct_tensor_onehot(dev_pos, alphabet_pos, dim_pos)
# 	dev_chunk_v = construct_tensor_onehot(dev_chunk, alphabet_chunk, dim_chunk)
# 	dev_tag_v = construct_tensor_onehot(dev_tag, alphabet_tag, dim_tag)

# 	test_pos_v = construct_tensor_onehot(test_pos, alphabet_pos, dim_pos)
# 	test_chunk_v = construct_tensor_onehot(test_chunk, alphabet_chunk, dim_chunk)
# 	test_tag_v = construct_tensor_onehot(test_tag, alphabet_tag, dim_tag)
	
# 	input_train = train_word_v
# 	input_train = np.concatenate((input_train, train_pos_v), axis=2)
# 	input_train = np.concatenate((input_train, train_chunk_v), axis=2)
# 	output_train = train_tag_v

# 	input_dev = dev_word_v
# 	input_dev = np.concatenate((input_dev, dev_pos_v), axis=2)
# 	input_dev = np.concatenate((input_dev, dev_chunk_v), axis=2)
# 	output_dev = dev_tag_v

# 	input_test = test_word_v
# 	input_test = np.concatenate((input_test, test_pos_v), axis=2)
# 	input_test = np.concatenate((input_test, test_chunk_v), axis=2)
# 	output_test = test_tag_v

# 	return input_train, output_train, input_test, output_test

def str_to_vec2():
	global input_train, output_train, input_test, output_test, input_train_add, input_test_add, input_dev, input_dev_add, output_dev

	dim_pos = alphabet_pos.size()
	dim_tag = alphabet_tag.size()

	train_word_v = word_to_index(train_word)
	train_pos_v = construct_tensor_onehot(train_pos, alphabet_pos, dim_pos)
	train_tag_v = construct_tensor_onehot(train_tag, alphabet_tag, dim_tag)

	dev_word_v = word_to_index(dev_word)
	dev_pos_v = construct_tensor_onehot(dev_pos, alphabet_pos, dim_pos)
	dev_tag_v = construct_tensor_onehot(dev_tag, alphabet_tag, dim_tag)

	test_word_v = word_to_index(test_word)
	test_pos_v = construct_tensor_onehot(test_pos, alphabet_pos, dim_pos)
	test_tag_v = construct_tensor_onehot(test_tag, alphabet_tag, dim_tag)

	input_train = train_word_v
	input_train_add = train_pos_v
	output_train = train_tag_v

	input_dev = dev_word_v
	input_dev_add = dev_pos_v
	output_dev = dev_tag_v

	input_test = test_word_v
	input_test_add = test_pos_v
	output_test = test_tag_v



def load_embedding_matrix():
	global embedd_words
	global embedd_vectors
	global unknown_embedd
	global unknown_embedd_pos
	global embedd_dim
	global embedd_matrix
	global zero_embedding_pos

	word_dir = '../vie-ner-lstm/embedding/words.pl'
	vector_dir = '../vie-ner-lstm/embedding/vectors.npy'
	embedd_vectors = np.load(vector_dir)
	with open(word_dir, 'rb') as handle:
		embedd_words = pickle.load(handle)

	embedd_dim = np.shape(embedd_vectors)[1]
	unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])
	embedd_matrix = np.zeros((len(embedd_words) + 2, embedd_dim))

	for i in range(len(embedd_vectors)):
		embedd_matrix[i] = embedd_vectors[i]

	embedd_matrix[-2] = unknown_embedd
	embedd_matrix[-1] = np.zeros((embedd_dim))
	unknown_embedd_pos = len(embedd_words)
	zero_embedding_pos = len(embedd_vectors) + 1

	
# def create_data():
#	load_embedding()
# 	load_data()
# 	str_to_id()
# 	str_to_vec()
# 	return input_train, output_train, input_test, output_test, alphabet_tag, embedd_matrix

def create_data(train_files, dev_files, test_files):
	load_data(train_files, dev_files, test_files)
	load_embedding_matrix()
	str_to_id()
	str_to_vec2()
	return input_train, input_train_add, input_dev, input_dev_add, input_test, input_test_add, output_train, output_dev, output_test, alphabet_tag, embedd_matrix

if __name__ == "__main__":
	load_embedding_matrix()
	# load_embedding()
	# load_data()
	# str_to_id()
	# str_to_vec()
	
	

