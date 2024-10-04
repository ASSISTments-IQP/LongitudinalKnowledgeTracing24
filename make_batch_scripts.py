if __name__ == '__main__':
	years = ['19-20','20-21','21-22','22-23','23-24']
	models = ['DKT', 'SAKT']
	fold_nums = range(5)
	sample_nums = range(1,11)

	for m in models:
		for y in years:
			for f in fold_nums:
				with open('./within_year_temp.sh', 'rt') as fin:
					with open (f'./job_start_scripts/{m}_{y}_{f}_wy.sh', 'wt') as fout:
						for line in fin:
							line = line.replace('MODEL_TYPE', m)
							line = line.replace('FOLD_NUM',str(f))
							line = line.replace('YEAR', y)
							fout.write(line)
			if y != '23-24':
				for s in sample_nums:
					with open('./cross_year_temp.sh', 'rt') as fin:
						with open(f'./job_start_scripts/{m}_{y}_{f}_cy.sh', 'wt') as fout:
							for line in fin:
								line = line.replace('MODEL_TYPE', m)
								line = line.replace('SAMPLE_NUM', str(s))
								line = line.replace('YEAR', y)
								fout.write(line)
