import pickle
import os
import nltk
import re
from config import Configs
from shutil import copyfile
import subprocess
import pathlib


def to_str(c):
	s = ''
	for char in c:
		s += char[0]
	return s


############################################################################
# c[i] = [character, tag1, tag2]
# example:	c[23] = ['P', 3, 1]
############################################################################
def raw_to_character(filename):
	s = open(filename, 'r').read() # read all file content to a string
	c = [[s[i]] for i in range(len(s))] # init character list with no tags

	# first mark all tags in s
	for tag in Configs.tags:
		locs = [(m.start(), m.end()) for m in re.finditer(re.escape(tag), s)]
		val = Configs.tags[tag]
		for (start, end) in locs:
			for i in range(start, end):
				c[i].append(val)

	# stack represent current tag for a character
	# for example: curr = [0, -3, -1] => <org>...<per>..(current_character)..</per>..<org>
	st = [0] 
	for i in range(len(s)):
		if len(c[i]) == 1: # normal character
			c[i] = [c[i][0]] + [-i for i in st]
			continue
		if c[i][1] == -10:
			if c[i][0] == '<':
				st = st[:-1] # pop stack, if first character of <endtag>
			continue
		if c[i][1] < 0:
			if c[i][0] == '<':
				st.append(c[i][1]) # push tag to stack, if first character of <tag>
			continue		

	# remove tags
	result = []
	for char in c:
		if char[1] >= 0:
			del char[1]
			while len(char) < 3: # fill by O tag
				char.append(0)
			result.append(char)
	return result


############################################################################
# t[i] = [token, index, tag1, tag2]
# example:	t[101] = ['Nam', 1239, 3, 4]
############################################################################
def token_mapping(c, s, tokens):
	# match tokens with position in c
	i = 0
	t = []
	for token in tokens:		
		# nltk auto convert (") to (``) and ('') , however ('') token is not changed !!
		if (token == '``' or token == "''") and not "''" in s[i:i+20]: 
			token = '"'
		m = len(token)
		while not fuzzy_cmp(s[i:i + m], token):
			i += 1
		t.append([token, i] + c[i][1:])
		i += m
	return t

def token_mapping2(c, s, tokens): # handle inconsistent tokenizer
	# match tokens with position in c
	i = 0
	t = []

	for token in tokens:		
		# nltk auto convert (") to (``) and ('') , however ('') token is not changed !!
		if (token == '``' or token == "''") and not "''" in s[i:i+20]: 
			token = '"'
		m = len(token)
		if token == '':		
			t.append([token, i] + [0,0])
			continue
		while s[i].strip() == '':
			i += 1
		t.append([token, i] + c[i][1:])
		i += m
	return t


def all_raw_to_character():
	# print('Reading all data to folder: ' + folder)
	result = []
	for filename in scan_data():
		result += raw_to_character(filename)
	return result


def scan_data():
	for dir1 in os.listdir("../data"):
		if not os.path.exists("../out/" + dir1):
			os.makedirs("../out/" + dir1)
		for dir2 in os.listdir("../data/" + dir1):
			if not os.path.exists("../out/" + dir1 + "/" + dir2):
				os.makedirs("../out/" + dir1 + "/" + dir2)
			for file in os.listdir("../data/" + dir1 + "/" + dir2):
				yield '../data/' + dir1 + "/" + dir2 + "/" + file


def tokenize_nltk():
	nltk.download('punkt')
	c = all_raw_to_character()
	s = to_str(c)
	tokens = token_mapping2(c, s, nltk.word_tokenize(s))
	return write_token_to_file(tokens, '../data_nltk/')


def tokenize_vncorenlp(file):
	if os.path.isfile(file.replace('data', 'data_conll')):
		print(file)
		return

	tmp_inp = '../tmp/in'
	tmp_out = '../tmp/out'
	c = raw_to_character(file)
	s = to_str(c)
	f = open(tmp_inp, 'w')
	f.write(s)
	f.close()
	
	cmd = 'cd ../VnCoreNLP2/ ;'
	cmd += '/usr/java/jre1.8.0_161/bin/java -Xmx2g -jar VnCoreNLP-1.0.jar '
	cmd += '-fin ../tmp/in '
	cmd += '-fout ../tmp/out -annotators wseg,pos'
	print(subprocess.check_output(cmd, shell=True))

	f = open(tmp_out, 'r')
	tokens = []
	lines = []
	for line in f:
		if line.strip() == '': # end of sentence
			lines.append([0] * 6)
			lines[-1][1] = ''
		else:
			lines.append(line.split('\t'))
		tokens.append(lines[-1][1].replace('_', ' '))
	t = token_mapping2(c, to_str(c), tokens)	

	file = file.replace('data', 'data_conll').replace(".txt", ".muc")
	directory = os.path.dirname(file)
	pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


	f_dat = open(file, 'w')
	f_idx = open(file.replace('muc', 'idx'), 'w')
	(prev_tag1, prev_tag2) = ('O', 'O')
	for i in range(len(t)):
		# print(t[i])
		if t[i][0] != '':
			tag1 = Configs.conll_tag[t[i][2]]
			tag2 = Configs.conll_tag[t[i][3]]
			tmp = (tag1, tag2)
			if tag1 != "O":
				if tag1 != prev_tag1:
					tag1 = "B-" + tag1
				else:
					tag1 = "I-" + tag1
			if tag2 != "O":
				if tag2 != prev_tag2:
					tag2 = "B-" + tag2
				else:
					tag2 = "I-" + tag2
			(prev_tag1, prev_tag2) = tmp

			dat = lines[i][1] + "\t" + lines[i][2] + "\tO\t" + tag1 + "\t" + tag2 + "\n"
			f_dat.write(dat)
			f_idx.write("%i\n" % t[i][1])
		else:
			(prev_tag1, prev_tag2) = ('O', 'O')
			f_dat.write('\n')
			f_idx.write('\n')
	f_dat.close()
	f_idx.close()

# def tokenize_nnvlp(file):


def tokenize():
	files = scan_data()
	for file in files:
		tokenize_vncorenlp(file)


def get_data(t, topics):
	cmd = "find ../data_conll/%s" % t
	files = str(subprocess.check_output(cmd, shell=True)).split("\\n")
	dat = ''
	idx = ''
	for file in files:
		for topic in topics:
			if not topic in file:
				continue
			if ".muc" in file:
				dat += open(file, "r").read()
			if ".idx" in file:
				idx += open(file, "r").read()
			break
	return dat, idx


def group_file_to_topic():
	# topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
	# 	"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
	topics = ["Cong_nghe","Giai_tri","Khoa_hoc","Nha_dat","The_gioi","Van_hoa",\
	"Doi_song","Giao_duc","Kinh_te","Phap_luat","The_thao","Xa_hoi"]

	for topic in topics:
	# 	dat, idx = get_data("Train", [topic])
	# 	f = open("../data_conll_topic/Train/%s.muc" % topic, "w")
	# 	f.write(dat)
	# 	f.close()

	# 	dat, idx = get_data("Dev", [topic])
	# 	f = open("../data_conll_topic/Dev/%s.muc" % topic, "w")
	# 	f.write(dat)
	# 	f.close()

		dat, idx = get_data("Test", [topic])
		f = open("../data_conll_topic/Test/%s.muc" % topic, "w")
		f.write(dat)
		f.close()




if __name__ == '__main__':
	# tokenize()
	# get_data("Test", ["Doi_song"])
	group_file_to_topic()





	

				

