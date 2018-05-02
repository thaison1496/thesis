import subprocess

# topics = ["Doi_song","Giao_duc","Kinh_te","The_gioi",\
# 		"Van_hoa","Giai_tri","KH-CN","Phap_luat","The_thao","Xa_hoi"]

# topics = ["KH-CN","Phap_luat","The_thao","Xa_hoi"]

# w = 0.1
# for k in range(11):
# 	cmd = "python tests.py %s | tee ../log/vanhoa_weigth.%f" % (k*w, k*w)
# 	print(cmd)
# 	subprocess.check_output(cmd, shell=True)


for in_domain in range(5, 6):
	cmd = "python tests.py %d | tee ../log/kinhte_indomain.%d" % (in_domain, in_domain * 250)
	print(cmd)
	subprocess.check_output(cmd, shell=True)