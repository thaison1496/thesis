import ner
import sys

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


# 2449/2449 [==============================] - 20s 8ms/step - loss: 0.0292 - acc: 0.9907 - val_loss: 0.0176 - val_acc: 0.9952
# Testing model...
# ['B-PER', 'O', 'B-LOC', 'B-ORG', 'I-ORG', 'I-PER', 'B-MIC', 'I-MIC', 'I-LOC']
# (817, 136)
# processed 21900 tokens with 975 phrases; found: 890 phrases; correct: 665.
# accuracy:  97.79%; precision:  74.72%; recall:  68.21%; FB1:  71.31
#               LOC: precision:  83.16%; recall:  77.07%; FB1:  80.00  291
#               MIC: precision:   0.00%; recall:   0.00%; FB1:   0.00  28
#               ORG: precision:  72.46%; recall:  60.45%; FB1:  65.91  443
#               PER: precision:  79.69%; recall:  78.46%; FB1:  79.07  128
# Running time: 
# 0:16:07.684321
def kinh_te():
	train_files = ["../folds/fold_1/Kinh_te.train"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 50)


# Epoch 135/500
# 2449/2449 [==============================] - 20s 8ms/step - loss: 0.0068 - acc: 0.9978 - val_loss: 0.0018 - val_acc: 0.9995
# processed 21900 tokens with 938 phrases; found: 890 phrases; correct: 680.
# accuracy:  97.93%; precision:  76.40%; recall:  72.49%; FB1:  74.40
#               LOC: precision:  84.88%; recall:  76.00%; FB1:  80.19  291
#               MIC: precision:   3.57%; recall:  25.00%; FB1:   6.25  28
#               ORG: precision:  73.36%; recall:  68.13%; FB1:  70.65  443
#               PER: precision:  83.59%; recall:  81.06%; FB1:  82.31  128
# Testing model...
# ['B-PER', 'O', 'B-LOC', 'B-ORG', 'I-ORG', 'I-PER', 'B-MIC', 'I-MIC', 'I-LOC']
# (817, 136)
# Running time: 
# 0:49:24.697321
def kinh_te2():
	train_files = ["../folds/fold_1/Kinh_te.train"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 500)


# Epoch 125/500
# 2449/2449 [==============================] - 19s 8ms/step - loss: 0.0075 - acc: 0.9978 - val_loss: 0.0024 - val_acc: 0.9993
# processed 21900 tokens with 963 phrases; found: 862 phrases; correct: 696.
# accuracy:  98.21%; precision:  80.74%; recall:  72.27%; FB1:  76.27
#               LOC: precision:  84.88%; recall:  75.54%; FB1:  79.94  291
#               ORG: precision:  77.20%; recall:  67.06%; FB1:  71.77  443
#               PER: precision:  83.59%; recall:  84.92%; FB1:  84.25  128

def kinh_te_no_mic():
	train_files = ["../folds/fold_1/Kinh_te.train"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 500)


# Epoch 135/500
# 2449/2449 [==============================] - 41s 17ms/step - loss: 0.0059 - acc: 0.9980 - val_loss: 9.2868e-04 - val_acc: 0.9998
# processed 21900 tokens with 946 phrases; found: 862 phrases; correct: 718.
# accuracy:  98.31%; precision:  83.29%; recall:  75.90%; FB1:  79.42
#               LOC: precision:  88.32%; recall:  80.06%; FB1:  83.99  291
#               ORG: precision:  79.23%; recall:  68.96%; FB1:  73.74  443
#               PER: precision:  85.94%; recall:  94.83%; FB1:  90.16  128
# Testing model...
# ['B-PER', 'O', 'B-LOC', 'B-ORG', 'I-ORG', 'I-PER', 'I-LOC']
# (817, 136)
# Running time: 
# 1:40:16.656015

# real	100m19.794s
# user	159m47.853s
# sys	7m3.040s
def kinh_te_2_lstm():
	train_files = ["../folds/fold_1/Kinh_te.train"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 500)


def topic_2_lstm(topic):
	train_files = ["../folds/fold_1/%s.train" % topic]
	dev_files = ["../folds/fold_1/%s.dev" % topic]
	test_files = ["../folds/fold_1/%s.test" % topic]
	ner.run(train_files, dev_files, test_files, 500, name=topic)


# test transfer model correctness
def kinh_te_transfer_model():
	train_files = ["../folds/fold_1/Kinh_te.train"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 500)


def kinhte_thethao_outdomain():
	train_files = ["../folds/fold_1/Kinh_te.train"]
	dev_files = ["../folds/fold_1/The_thao.dev"]
	test_files = ["../folds/fold_1/The_thao.test"]
	ner.run(train_files, dev_files, test_files, 500)


def kinhte_thethao_mix():
	train_files = ["../folds/fold_1/The_thao.train",
		"../folds/fold_1/Kinh_te.train",
		"../folds/fold_1/Kinh_te.dev",
		"../folds/fold_1/Kinh_te.test"]
	dev_files = ["../folds/fold_1/The_thao.dev"]
	test_files = ["../folds/fold_1/The_thao.test"]
	ner.run(train_files, dev_files, test_files, 500)


def kinhte_thethao_transfer():
	train_files = [("../folds/fold_1/The_thao.train", 1),
		("../folds/fold_1/Kinh_te.train", 0),
		("../folds/fold_1/Kinh_te.dev", 0),
		("../folds/fold_1/Kinh_te.test", 0)]
	dev_files = [("../folds/fold_1/The_thao.dev", 1)]
	test_files = [("../folds/fold_1/The_thao.test", 1)]
	ner.run(train_files, dev_files, test_files, 500)


def thethao_kinhte_outdomain():
	train_files = ["../folds/fold_1/The_thao.train"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 500)


def thethao_kinhte_mix():
	train_files = ["../folds/fold_1/Kinh_te.train",
		"../folds/fold_1/The_thao.train",
		"../folds/fold_1/The_thao.dev",
		"../folds/fold_1/The_thao.test"]
	dev_files = ["../folds/fold_1/Kinh_te.dev"]
	test_files = ["../folds/fold_1/Kinh_te.test"]
	ner.run(train_files, dev_files, test_files, 500)


def thethao_kinhte_transfer():
	print("thethao_kinhte_transfer")
	train_files = [("../folds/fold_1/Kinh_te.train", 1),
		("../folds/fold_1/The_thao.train", 0),
		("../folds/fold_1/The_thao.dev", 0),
		("../folds/fold_1/The_thao.test", 0)]
	dev_files = [("../folds/fold_1/Kinh_te.dev", 1)]
	test_files = [("../folds/fold_1/Kinh_te.test", 1)]
	ner.run(train_files, dev_files, test_files, 500)


def kinhte_no_shuffle():
	# topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
	# 	"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
	topics = ["Kinh_te"]
	train_files = []
	dev_files = []
	test_files = []
	for topic in topics:
		train_files.append("../data_conll_topic/News_Train/%s.muc" % topic)
		dev_files.append("../data_conll_topic/News_Dev/%s.muc" % topic)
		test_files.append("../data_conll_topic/News_Test/%s.muc" % topic)
	ner.run(train_files, dev_files, test_files, 500)


def all_mix():
	print("all_mix")
	train_files = []
	dev_files = []
	test_files = []
	topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
		"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]
	for topic in topics:
		train_files.append("../folds/fold_1/%s.train" % topic)
		dev_files.append("../folds/fold_1/%s.dev" % topic)
		test_files.append("../folds/fold_1/%s.test" % topic)
	ner.run(train_files, dev_files, test_files, 500)

	
if __name__ == "__main__":
	# ner_topic_pair_test()
	# news_1()
	# news_1_no_val()
	# kinh_te2()
	# kinh_te_no_mic()
	# kinh_te_2_lstm()
	# topic_2_lstm(sys.argv[1].strip())
	# topic_2_lstm(sys.argv[1])
	# kinhte_thethao()
	# kinhte_thethao_outdomain()
	# kinhte_thethao_mix()
	# kinhte_thethao_transfer()
	# kinhte_no_shuffle()
	# thethao_kinhte_outdomain()
	# thethao_kinhte_mix()
	# all_mix()
	thethao_kinhte_transfer()
	
