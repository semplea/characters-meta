# coding: utf8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plotJobScores(similarity_scores):
	"""
	Create plot of job score results
	"""
    # white backround for report
	palette = sns.color_palette()
	sns.set_style('whitegrid')
	fig, ax = plt.subplots()
	sns.swarmplot(
		x='Character',
		y='Similarity',
		data=similarity_scores,
		hue='Predictor',
		palette={'Count': palette[0], 'Proximity': palette[2]},
		ax=ax)
	plt.show()

	fig, ax = plt.subplots()
	sns.swarmplot(
		x='Rank',
		y='Similarity',
		data=similarity_scores,
		hue='Predictor',
		palette={'Count': palette[0], 'Proximity': palette[2]},
		ax=ax)
	plt.show()

	# add scatter plot for rank to similarity score
