# coding: utf8
#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec

data = {'Mr. A':[1,0,3,3,0,6,0],'Mme B':[1,0,0,0,5,2,2],'C':[0,0,4,5,0,0,0]}
places = ['place 1', 'bb', 'cdef', 'place 4', 'world', 'nowhere', 'x']

# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
				  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
				  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
				  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# You typically want your plot to be ~1.33x wider than tall. This plot
# is a rare exception because of the number of lines being plotted on it.
# Common sizes: (10, 7.5) and (12, 9)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Remove the plot frame lines. They are unnecessary here.
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
plt.xlim(1980.5, 2011.1)
plt.ylim(0, 5)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(range(1, 6, 1), fontsize=14)
plt.yticks(range(1, len(places), 1), places, fontsize=14)

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
'''
for y in range(10, 91, 10):
	plt.plot(range(1969, 2012), [y] * len(range(1969, 2012)), '--', lw=0.5, color='black', alpha=0.3)
'''
# Remove the tick marks; they are unnecessary with the tick lines we just
# plotted.
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='on', left='off', right='off', labelleft='on')

chars = data.keys()

for rank, column in data.iteritems():
	print column
	# Plot each line separately with its own color.

	line = plt.plot(places, column, lw=2.5, color=color_sequence[rank])

	# Add a text label to the right end of every line. Most of the code below is adding specific offsets y position because some labels overlapped.
	y_pos = data[column][-1] - 0.5

	# Again, make sure that all labels are large enough to be easily read by the viewer.
#	plt.text(2011.5, y_pos, column, fontsize=14, color=color_sequence[rank])

# Make the title big enough so it spans the entire plot, but don't make it so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include axis labels; they are self-evident, in this plot's case.
plt.title('Storyline interpretation: places vs. characters\n', fontsize=18, ha='center')
plt.show()
#plt.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')		// change extension for jpeg or pdf
