import subprocess

topics = ["Doi_song","Giao_duc", "The_gioi",\
		"Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]

# topics = ["KH-CN","Phap_luat","The_thao","Xa_hoi"]

# w = 0.1
# for k in range(11):
# 	cmd = "python tests.py %s | tee ../log/vanhoa_weigth.%f" % (k*w, k*w)
# 	print(cmd)
# 	subprocess.check_output(cmd, shell=True)


for src in topics:
	# cmd = "python tests.py %s | tee ../log/%s_kinhte_mixin" % (src, src)
	cmd = "python tests.py %s | tee ../log/%s_kinhte_w_0.3" % (src, src)
	print(cmd)
	subprocess.check_output(cmd, shell=True)