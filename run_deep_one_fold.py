from DKT.DKT_pt import DKT
from SAKT.sakt_pt import SAKTModel
import pandas as pd
import sys, json

model_list = ['DKT', 'SAKT-E', 'SAKT-KC']


def run_one_sample(train, test_samps, model_type):
	if model_type == 'DKT':
		model = DKT(16, 40, 96, 2e-3, 0.278, 1.4e-5)
		num_epochs = 100
	elif model_type == 'SAKT-E':
		model = SAKTModel(60, 64, 352, 8, 0.43, 1e-4, 0.7, feature_col='old_problem_id')
		num_epochs = 23
	else:
		model = SAKTModel(100, 48, 128, 16, 0.188, 1e-4, 0.868, feature_col='skill_id')
		num_epochs = 25
	model.fit(train, num_epochs)
	res = {}
	for year, samp in test_samps.items():
		eval_tup = model.evaluate(samp)
		res[year] = {
			'auc': eval_tup[0],
			'll': eval_tup[1],
			'f1': eval_tup[2]
		}
	return res

if __name__ == '__main__':
	year_list = ['19-20', '20-21', '21-22', '22-23', '23-24']
	if len(sys.argv) != 4:
		raise Exception("Invalid number of args specified")

	model_type = sys.argv[1]
	train_year = sys.argv[2]
	sample_num = int(sys.argv[3])

	if model_type not in model_list:
		raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")
	elif train_year not in year_list:
		raise Exception(f'Invalid year specified, model year must be one of {str(year_list)[1:-1]}')
	elif sample_num not in range(1, 11):
		raise Exception(f"Invalid sample number provided")

	years = ['19-20', '20-21', '21-22', '22-23', '23-24']
	sample_dict = {}
	print('Loading year samples')

	train_dict = {}
	for i in range(1, 11):
		s1 = pd.read_csv(f'../Data/samples/{train_year}/sample{i}.csv')
		train_dict[i] = s1

	train_sample = train_dict.pop(sample_num)
	wy_test = pd.concat(train_dict)
	test_years = year_list[year_list.index(train_year):]


	test_samps = {}
	test_samps[train_year] = wy_test
	for y in test_years:
		test_samps[y] = pd.read_csv(f'../Data/samples/{y}/sample{sample_num}.csv')

	res = run_one_sample(train_sample, test_samps, model_type)

	with open(f'./{model_type}_{train_year}_{sample_num}.json', 'w') as fout:
		json.dump(res, fout)
		fout.close()
