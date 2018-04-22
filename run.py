import subprocess

topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
		"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]

# topics = ["Phap_luat","The_thao","Xa_hoi"]

for topic in topics:
	cmd = "python tests.py %s | tee ../log/%s_no_shuffle" % (topic, topic)
	print(cmd)
	subprocess.check_output(cmd, shell=True)