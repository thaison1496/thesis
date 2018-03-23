import gen_feature

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


if __name__ == "__main__":
	ner_topic_pair_test()
