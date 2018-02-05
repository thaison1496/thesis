class Configs:
	tags = {
		'<ENAMEX TYPE="PERSON">': -1,
		'<ENAMEX TYPE="LOCATION">': -2,
		'<ENAMEX TYPE="ORGANIZATION">': -3,
		'<ENAMEX TYPE="MISCELLANEOUS">': -4,
		'</ENAMEX>': -10,
		'<ENAMEX TYPE="O">': 0
	}	
	inv_tags = {v: k for k, v in tags.items()}
	conll_tag = {
		1: 'PER',
		2: 'LOC',
		3: 'ORG',
		4: 'MIC',
		0: 'O'
	}
	# compact_inv_tags = {}

# Configs.compact_inv_tags = {
# 	k: Configs.inv_tags[k].replace('<ENAMEX TYPE="','').replace('">','')[:3] for k, v in Configs.inv_tags.items()
# }
