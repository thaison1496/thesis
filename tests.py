import ner

# with every pair of topic, run maxent model with 1 topic as train and the other as test
def ner_topic_pair_test():
	topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
		"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]

	# result[t_train][t_dev] = (precision, recall, f1)
	result = {k: [] for k in topics}

	for t_train in topics:
		for t_dev in topics:
			if t_dev == t_train:
				continue
			print("Evaluating:\nTrain: %s\nTest: %s" % (t_train, t_dev))


def t1(): # old
	topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
		"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
	train_files = []
	test_files = []
	for topic in topics:
		train_files.append("../data_conll_topic/Train/%s.muc" % topic)
		test_files.append("../data_conll_topic/Dev/%s.muc" % topic)
	ner.run(train_files, test_files, 50)


def social_1():
	# topics = ["Cong_nghe","Giao_duc","Nha_dat","The_thao","Doi_song","Khoa_hoc","Phap_luat","Van_hoa","Giai_tri","Kinh_te","The_gioi","Xa_hoi"]
	topics = ["Cong_nghe","Giao_duc","Nha_dat","The_thao","Doi_song","Khoa_hoc","Phap_luat"]
	# topics = ["Cong_nghe", "Kinh_te"]
	train_files = []
	dev_files = []
	test_files = []
	for topic in topics:
		train_files.append("../data_conll_topic/Social_Train/%s.muc" % topic)
		dev_files.append("../data_conll_topic/Social_Dev/%s.muc" % topic)
		test_files.append("../data_conll_topic/Social_Test/%s.muc" % topic)
	ner.run(train_files, dev_files, test_files, 100)


# 2733/2733 [==============================] - 30s 11ms/step - loss: 0.0336 - acc: 0.9895 - val_loss: 0.0197 - val_acc: 0.9947
# Testing model...
# ['O', 'B-ORG', 'B-PER', 'I-ORG', 'I-PER', 'B-LOC', 'B-MIC', 'I-MIC', 'I-LOC']
# (920, 131)
# processed 22111 tokens with 1108 phrases; found: 1043 phrases; correct: 781.
# accuracy:  97.52%; precision:  74.88%; recall:  70.49%; FB1:  72.62
#               LOC: precision:  83.38%; recall:  77.98%; FB1:  80.59  361
#               MIC: precision:  16.67%; recall:  18.52%; FB1:  17.54  30
#               ORG: precision:  74.10%; recall:  69.11%; FB1:  71.52  471
#               PER: precision:  69.61%; recall:  66.32%; FB1:  67.92  181
# Running time: 
# 0:25:35.615248
def news_1():
	# topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
	# 	"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
	topics = ["KH-CN", "Kinh_te"]
	train_files = []
	dev_files = []
	test_files = []
	for topic in topics:
		train_files.append("../data_conll_topic/News_Train/%s.muc" % topic)
		dev_files.append("../data_conll_topic/News_Dev/%s.muc" % topic)
		test_files.append("../data_conll_topic/News_Test/%s.muc" % topic)
	ner.run(train_files, dev_files, test_files, 50)


# 2733/2733 [==============================] - 20s 7ms/step - loss: 0.0347 - acc: 0.9889
# Testing model...
# ['O', 'B-ORG', 'B-PER', 'I-ORG', 'I-PER', 'B-LOC', 'B-MIC', 'I-MIC', 'I-LOC']
# (920, 131)
# processed 22111 tokens with 1133 phrases; found: 1043 phrases; correct: 783.
# accuracy:  97.52%; precision:  75.07%; recall:  69.11%; FB1:  71.97
#               LOC: precision:  84.21%; recall:  76.00%; FB1:  79.89  361
#               MIC: precision:  23.33%; recall:  26.92%; FB1:  25.00  30
#               ORG: precision:  72.82%; recall:  67.12%; FB1:  69.86  471
#               PER: precision:  71.27%; recall:  65.82%; FB1:  68.44  181
# Running time: 
# 0:18:40.887292
def news_1_no_val():
	# topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
	# 	"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
	topics = ["KH-CN", "Kinh_te"]
	train_files = []
	dev_files = []
	test_files = []
	for topic in topics:
		train_files.append("../data_conll_topic/News_Train/%s.muc" % topic)
		dev_files.append("../data_conll_topic/News_Dev/%s.muc" % topic)
		test_files.append("../data_conll_topic/News_Test/%s.muc" % topic)
	ner.run(train_files, dev_files, test_files, 50, no_val=True)


if __name__ == "__main__":
	# ner_topic_pair_test()
	# news_1()
	news_1_no_val()
