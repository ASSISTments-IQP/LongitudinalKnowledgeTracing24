if __name__ == '__main__':
	years = ['19-20','20-21','21-22','22-23','23-24']
	models = ['BKT', 'PFA']

	for m in models:
		for y in years:
			with open(f'within_year_shallow_temp.sh', 'rt') as fin:
				with open (f'./job_start_scripts/{m}_{y}.sh', 'wt') as fout:
					for line in fin:
						line = line.replace('MODEL_TYPE', m)
						line = line.replace('YEAR', y)
						fout.write(line)
