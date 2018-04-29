import codecs
from collections import OrderedDict

def read_conll_format(topics):
	sum_sent_len = 0
	num_sent = 0
	num_token = 0
	for topic in topics:
		# print(topic)
		ne = OrderedDict()
		for e in ("PER", "LOC", "ORG"):
				ne[e] = []
		for t in ("Train", "Test"):
			# print(t)
			
			input_file = "../data_conll_topic/News_%s/%s.muc" % (t, topic)
			with codecs.open(input_file, 'r', 'utf-8') as f:
				sent_len = 0
				for line in f:
					line = line.strip().split("\t")
					if (len(line) > 1):
						num_token += 1
						sent_len += 1
						word = line[0]
						tag = line[3].split("-")[-1]
						if tag in ne:
							ne[tag].append(word)
					else:
						num_sent += 1
						sum_sent_len += sent_len
						sent_len = 0
		# print("%.02f\t%d\t" % (sum_sent_len/num_sent, num_token), end = '')
		# for e in ne:
		# 	print("%d (%d)\t" % (len(ne[e]), len(set(ne[e]))), end='')
		print(num_sent)
		# print()
		sum_sent_len = 0
		num_sent = 0
		sent_len = 0
		num_token = 0
		

def read_conll_format2(topics):
	sum_sent_len = 0
	num_sent = 0
	num_token = 0

	c = 0
	s = 0
	for topic in topics:
		# print(topic)
		ne = OrderedDict()
		
		for t in ("Train", "Test"):
			ne[t] = OrderedDict()
			for e in ("PER", "LOC", "ORG"):
				ne[t][e] = []

			input_file = "../data_conll_topic/News_%s/%s.muc" % (t, topic)
			with codecs.open(input_file, 'r', 'utf-8') as f:
				sent_len = 0
				for line in f:
					line = line.strip().split("\t")
					if (len(line) > 1):
						num_token += 1
						sent_len += 1
						word = line[0]
						tag = line[3].split("-")[-1]
						if tag in ("PER", "LOC", "ORG"):
							ne[t][tag].append(word)

		
		for e in ("PER", "LOC", "ORG"):
			for word in ne["Test"][e]:
				if word in ne["Train"][e]:
					c += 1
			s += len(ne["Test"][e])
	print("%.02f" % (c / s), end="\t")
		
		


topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
		"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
read_conll_format(topics)
	