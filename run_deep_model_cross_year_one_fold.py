from PFA.PFA_Model import PFA
from bkt.BKT_Model import BKTModel
# from sakt.SAKT_model import SAKTModel
from DKT.DKT_Model import DKT
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json

model_list = ['DKT', 'SAKT']


def run_one_sample(train, test_samps, model_type):
	if model_type == 'SAKT':
		# model = SAKTModel()
		pass
	if model_type == 'DKT':
		model = DKT()
	model.fit(train)
	res = {}
	for year, samp in test_samps.items():
		res[year] = model.evaluate(samp)
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



	train_sample = pd.read_csv(f'../Data/samples/{train_year}/sample{sample_num}.csv')

	if train_year == '19-20':
		test_years = ['20-21', '21-22', '22-23', '23-24']
	elif train_year == '20-21':
		test_years = ['21-22', '22-23', '23-24']
	elif train_year == '21-22':
		test_years = ['22-23', '23-24']
	elif train_year == '22-23':
		test_years = ['23-24']
	else:
		raise Exception('Invalid training year provided')

	test_samps = {}
	for y in test_years:
		test_samps[y] = pd.read_csv(f'../Data/samples/{y}/sample{sample_num}.csv')

	res = run_one_sample(train_sample, test_samps, model_type)

	with open(f'./cy_{model_type}_{train_year}_{sample_num}.json', 'w') as fout:
		json.dump(res, fout)
		fout.close()
