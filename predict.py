import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, \
	classification_report
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def plot_roc_cur(f_per, t_per):
	plt.plot(f_per, t_per, color='orange', label='ROC')
	plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend()
	plt.show()


def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
	if not verbose:
		model.fit(X_train, y_train, verbose=0)
	else:
		model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	roc_auc = roc_auc_score(y_test, y_pred)
	coh_kap = cohen_kappa_score(y_test, y_pred)
	print("Accuracy = {}".format(accuracy))
	print("ROC Area under Curve = {}".format(roc_auc))
	print("Cohen's Kappa = {}".format(coh_kap))
	print(classification_report(y_test, y_pred, digits=5))

	probs = model.predict_proba(X_test)
	probs = probs[:, 1]
	f_per, t_per, thresholds = roc_curve(y_test, probs)
	# plot_roc_cur(f_per, t_per)
	plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='all')

	return model, accuracy, roc_auc, coh_kap


def get_result(world_cup, model, wmk, margin):
	world_cup['points'] = 0
	world_cup['total_prob'] = 0
	# print(world_cup)
	for group in set(world_cup['Group']):
		print('___Group {}:___'.format(group))
		for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
			print("{} vs {}: ".format(home, away), end='')
			row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
			home_rank = world_cup_rankings.loc[home, 'rank']
			home_points = world_cup_rankings.loc[home, 'total_points']
			opp_rank = world_cup_rankings.loc[away, 'rank']
			opp_points = world_cup_rankings.loc[away, 'total_points']
			row['average_rank'] = (home_rank + opp_rank) / 2
			row['rank_difference'] = home_rank - opp_rank
			row['point_difference'] = home_points - opp_points

			home_win = model.predict_proba(row)[:, 1][0]
			world_cup.loc[home, 'total_prob'] += home_win
			world_cup.loc[away, 'total_prob'] += 1 - home_win
			points = 0
			if home_win <= 0.5 - margin:
				print("{} wins with {:.2f}".format(away, 1 - home_win))
				world_cup.loc[away, 'points'] += 3
			if home_win > 0.5 - margin:
				points = 1
			if home_win >= 0.5 + margin:
				points = 3
				world_cup.loc[home, 'points'] += 3
				print("{} wins with {:.2f}".format(home, home_win))
			if points == 1:
				print("Draw")
				world_cup.loc[home, 'points'] += 1
				world_cup.loc[away, 'points'] += 1

	pairing = [0, 3, 4, 7, 8, 11, 12, 15, 1, 2, 5, 6, 9, 10, 13, 14]

	world_cup = world_cup.sort_values(by=['Group', 'points'], ascending=False).reset_index()
	next_round_wc = world_cup.groupby('Group').nth([0, 1])  # select the top 2
	next_round_wc = next_round_wc.reset_index()
	next_round_wc = next_round_wc.loc[pairing]
	next_round_wc = next_round_wc.set_index('Team')

	next_round_wc.plot.barh()
	plt.title('round of 16 points({})'.format(wmk))
	plt.tight_layout()
	plt.show()
	finals = ['Round_of_16', 'Quarterfinal', 'Semifinal', 'Final']

	labels = list()
	left = []
	right = []
	for f in finals:
		print("___{}___".format(f))
		iterations = int(len(next_round_wc) / 2)
		winners = []
		for i in range(iterations):
			home = next_round_wc.index[i * 2]
			away = next_round_wc.index[i * 2 + 1]
			print("{} vs {}: ".format(home, away), end='')
			row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
			home_rank = world_cup_rankings.loc[home, 'rank']
			home_points = world_cup_rankings.loc[home, 'total_points']
			opp_rank = world_cup_rankings.loc[away, 'rank']
			opp_points = world_cup_rankings.loc[away, 'total_points']
			row['average_rank'] = (home_rank + opp_rank) / 2
			row['rank_difference'] = home_rank - opp_rank
			row['point_difference'] = home_points - opp_points
			home_win = model.predict_proba(row)[:, 1][0]
			if model.predict_proba(row)[:, 1] <= 0.5:
				print("{0} wins with probability {1:.2f}".format(away, 1 - home_win))
				winners.append(away)
			else:
				print("{0} wins with probability {1:.2f}".format(home, home_win))
				winners.append(home)
			labels.append(
				"{} {}".format(world_cup_rankings.loc[home, 'country_abrv'],
							   world_cup_rankings.loc[away, 'country_abrv']))
			left.append(home_win)
			right.append(1 - home_win)
		next_round_wc = next_round_wc.loc[winners]
		print("\n")

	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.annotate('{:.2f}'.format(height),
						xy=(rect.get_x() + rect.get_width() / 2, height),
						xytext=(0, 1),  # 3点垂直偏移
						textcoords="offset points",
						ha='center', va='bottom')

	x = np.arange(8)
	width = 0.35
	fig, ax = plt.subplots()
	rects1 = ax.bar(x - width / 2, left[0:8], width)
	rects2 = ax.bar(x + width / 2, right[0:8], width)
	ax.set_ylabel('Probability')
	ax.set_xticks(x)
	ax.set_xticklabels(labels[0:8])
	ax.set_title('Round of 16({})'.format(wmk))
	fig.tight_layout()
	autolabel(rects1)
	autolabel(rects2)
	plt.show()

	x = np.arange(4)
	width = 0.35
	fig, ax = plt.subplots()
	rects3 = ax.bar(x - width / 2, left[8:12], width)
	rects4 = ax.bar(x + width / 2, right[8:12], width)
	ax.set_ylabel('Probability')
	ax.set_xticks(x)
	ax.set_xticklabels(labels[8:12])
	ax.set_title('Quarterfinal({})'.format(wmk))
	autolabel(rects3)
	autolabel(rects4)
	plt.show()

	x = np.arange(3)
	width = 0.35
	fig, ax = plt.subplots()
	rects5 = ax.bar(x - width / 2, left[12:], width)
	rects6 = ax.bar(x + width / 2, right[12:], width)
	ax.set_ylabel('Probability')
	ax.set_xticks(x)
	ax.set_xticklabels(labels[12:])
	ax.set_title('Semifinal and Final({})'.format(wmk))
	autolabel(rects5)
	autolabel(rects6)
	plt.show()


df = pd.read_csv('results.csv')
rank = pd.read_csv('fifa_ranking-2022-10-06.csv')
# 标准date
rank["rank_date"] = pd.to_datetime(rank["rank_date"])
df["date"] = pd.to_datetime(df["date"])
# 近三年的排名和比赛
rank = rank[(rank["rank_date"] >= "2019-1-1")].reset_index(drop=True)
df = df[(df["date"] >= "2019-1-1")].reset_index(drop=True)

rank["country_full"] = rank["country_full"].str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States")
rank = rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()

world_cup = pd.read_csv("Fifa_Worldcup_2022_Groups.csv")
# 替换国家名称
world_cup = world_cup.replace({"Korea Republic": "South Korea", "USA": "United States"})
world_cup = world_cup.set_index('Team')

df = df.merge(rank, left_on=['date', 'home_team'], right_on=['rank_date', 'country_full'])
df = df.merge(rank, left_on=['date', 'away_team'], right_on=['rank_date', 'country_full'], suffixes=('_home', '_away'))

df['rank_difference'] = df['rank_home'] - df['rank_away']
df['average_rank'] = (df['rank_home'] + df['rank_away']) / 2
df['point_difference'] = df['total_points_home'] - df['total_points_away']
df['score_difference'] = df['home_score'] - df['away_score']
df['is_won'] = df['score_difference'] > 0
df['is_stake'] = df['tournament'] != 'Friendly'

X, y = df.loc[:, ['average_rank', 'rank_difference', 'point_difference', 'is_stake']], df['is_won']

# 划分出训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)

# LogisticRegression
print("LogisticRegression: ")
params_lr = {'penalty': 'l1', 'solver': 'liblinear'}
model_lr = LogisticRegression(**params_lr)
model_lr, accuracy_lr, roc_auc_lr, coh_kap_lr = run_model(model_lr, X_train, y_train, X_test, y_test)

# 决策分类树
print("决策分类树: ")
params_dt = {'splitter': 'best', 'max_depth': 32, 'max_features': 'log2'}
model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt, coh_kap_dt = run_model(model_dt, X_train, y_train, X_test, y_test)

# MLPClassifier
print("MLPClassifier: ")
params_nn = {'hidden_layer_sizes': (30, 30, 30), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 500}
model_nn = MLPClassifier(**params_nn)
model_nn, accuracy_nn, roc_auc_nn, coh_kap_nn = run_model(model_nn, X_train, y_train, X_test, y_test)

# 随机森林
print("随机森林: ")
params_rf = {'n_estimators': 105, 'random_state': None, 'oob_score': True}
model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf = run_model(model_rf, X_train, y_train, X_test, y_test)

# lightgbm
print("lightgbm: ")
params_lgb = {
	'max_depth': 25,
	'min_split_gain': 0.1,
	'n_estimators': 1000,
	'num_leaves': 20,
	'reg_alpha': 1.2,
	'reg_lambda': 1.2,
	'subsample': 0.95,
	'subsample_freq': 20,
	'learning_rate': 0.05
}
model_lgb = lgb.LGBMClassifier(**params_lgb)
model_lgb, accuracy_lgb, roc_auc_lgb, coh_kap_lgb = run_model(model_lgb, X_train, y_train, X_test, y_test)
#
#
# XGBClassifier
print("XGBClassifier: ")
params_xgb = {'n_estimators': 1000, 'max_depth': 25, 'learning_rate': 0.05}
model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)

margin = 0.08
# let's define the rankings at the time of the World Cup
world_cup_rankings = rank.loc[
	(rank['rank_date'] == rank['rank_date'].max()) & rank['country_full'].isin(world_cup.index.unique())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])

opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']

world_cup['points'] = 0
world_cup['total_prob'] = 0

get_result(world_cup, model_lgb, 'lightgbm', margin)
get_result(world_cup, model_xgb, 'XGBClassifier', margin)
get_result(world_cup, model_rf, 'RandomForest', margin)
get_result(world_cup, model_nn, 'MLPClassifier', margin)
get_result(world_cup, model_dt, 'DecisionTreeClassifier', margin)
get_result(world_cup, model_lr, 'LogisticRegression', margin)
