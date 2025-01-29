if __name__ == '__main__':
	years = ['19-20','20-21','21-22','22-23','23-24']
	models = ['DKT', 'SAKT-KC', 'SAKT-E']
	sample_nums = range(1,11)

	for m in models:
		for y in years:
			for s in sample_nums:
				with open('deep_temp.sh', 'rt') as fin:
					with open (f'./job_start_scripts/{m}_{y}_{s}.sh', 'wt') as fout:
						for line in fin:
							line = line.replace('MODEL_TYPE', m)
							line = line.replace('FOLD_NUM',str(s))
							line = line.replace('YEAR', y)
							fout.write(line)

