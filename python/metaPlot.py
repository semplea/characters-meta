# coding: utf8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plotJobScores(similarity_scores):
	"""
	Create plot of job score results
	"""
    # white backround for report
    sns.set_style('whitegrid')
	palette = sns.color_palette()
	ax = sns.swarmplot(x='Character', y='Similarity', data=similarity_scores, hue='Predictor', palette={'Count': palette[0], 'Proximity': palette[2]})
	plt.show()
