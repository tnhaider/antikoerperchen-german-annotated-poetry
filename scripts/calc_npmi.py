import sys, csv
from time import sleep
from collections import Counter
from math import log2, log
from decimal import Decimal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
# sphinx_gallery_thumbnail_number = 2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from inout.utils.helper import *

def get_measures_emotions(infilename):
	alllabels = []
	allemotions = []
	allcombis = []
	tsv_file = open(infilename, 'r')
	lines = csv.reader(tsv_file, delimiter="\t")
	c = 0
	for line in lines:
		c += 1
		if c == 1:
			continue
		#if c > 5: 
		#	break
		emotions = line[12:16]
		measure = line[16]
		imeasure = line[17]
		enj = line[26]
		smeasure = line[18]
		meter = line[20]
		#author = line[7]
		period = line[9]
		measure = get_versification(meter, measure_type='s', greek_forms=False)
		#emotion1 = line[12]
		#emotion2 = line[13]
		#emotion3 = line[12]
		#emotion4 = line[13]
		#alllabels.append(period)
		#allemotions.append(measure)
		#allcombis.append((period, measure))
		#print(line)
		#print(emotions)
		#print(measure)
		#if emotion1 != 'NONE' and emotion2 != 'NONE':
		#allemotions.append(emotion1)
		#allemotions.append(emotion2)
		#allcombis.append((emotion1, emotion2))
		#if emotion3 != 'NONE' and emotion4 != 'NONE':
		#allemotions.append(emotion3)
		#allemotions.append(emotion4)
		#allcombis.append((emotion3, emotion4))
			
		ec = 0
		for emotion in emotions:
			ec += 1
			#if emotion == 'NONE':
			#	continue
			#if ec == 2 or ec == 4:
			#	continue
			allemotions.append(emotion)
			allcombis.append((measure, emotion))
		alllabels.append(measure)
		
		
		
	#return allcombis, allemotions, allmeasures
	return allcombis, allemotions, alllabels

def get_label_combis(infilename):
	alllabels = []
	allemotions = []
	allcombis = []
	tsv_file = open(infilename, 'r')
	lines = csv.reader(tsv_file, delimiter="\t")
	c = 0
	for line in lines:
		c += 1
		if c == 1:
			continue
		#if c > 5: 
		#	break
		emotions = line[12:16]
		measure = line[16]
		imeasure = line[17]
		enj = line[26]
		smeasure = line[18]
		meter = line[20]
		#author = line[7]
		measure = get_versification(meter, measure_type='f', greek_forms=True)
		period = line[9]
		periods = clean_period_label(period)
		#print(periods)
		#periods = re.sub(r"([A-Z])", r" \1", period).split()
		#if len(periods) == 1:
		#	periods = periods[0]
		#else:
		#	periods = periods[0] + periods[-1]
		#periods = ' // '.join(periods)
		alllabels.append(measure)
		allemotions.append(periods)
		allcombis.append((measure,periods))

	return allcombis, allemotions, alllabels

def clean_period_label(label):
	d = {'Barock':'Barock', 'Spätromantik':'Spätromantik', 'ArbeiterliteraturArbeiterdichtung, Expressionismus':'Arbeiterdichtung / Expressionismus', 'RomantikWeimarerKlassik':'Romantik / Weimarer Klassik', 'FrühexpressionismusArbeiterdichtung':'Frühexpressionismus / Arbeiterdichtung', 'Expressionismus':'Expressionismus', 'Symbolismus':'Symbolismus', 'Realismus':'Realismus', 'VormärzBiedermeierJunges Deutschland, ArbeiterliteraturArbeiterdichtung':'Vormärz / Junges Deutschland', 'Biedermeier':'Biedermeier', 'SymbolismusExpressionismus':'Symbolismus / Expressionismus', 'Sturm und DrangGeniezeit':'Sturm und Drang', 'Romantik':'Romantik', 'Literatur im NationalsozialismusExilliteraturEmigrantenliteratur':'Exilliteratur / Literatur d. Nationalsozialismus', 'Weimarer Klassik':'Weimarer Klassik', 'VormärzBiedermeierJunges Deutschland':'Vormärz / Junges Deutschland', 'Empfindsamkeit':'Empfindsamkeit', 'Heidelberger RomantikJüngere RomantikHochromantik':'Hochromantik / Heidelberger Romantik', 'Naturalismus':'Naturalismus', 'FrühromantikÄltere Romantik':'Frühromantik'}
	return d[label]


def get_probs(counter_dict):
	prob_dict = {}
	val_sum = float(sum(counter_dict.values()))
	for key, value in counter_dict.items():
		prob_dict[key] = value / val_sum
	return prob_dict

def calc_npmi(joint, X, Y):
	outdict = {}
	for x in X:
		for y in Y:
			try:
				pxy = joint[(x, y)]
			except KeyError:
				pxy = 0
				outdict[(x, y)] = -1
				continue
			px = float(X[x])
			py = float(Y[y])
			pxpy = px * py
			pmi = log2(pxy) - log2(pxpy)

			hxy = -log2(pxy)

			npmi = pmi/hxy

			outdict[(x,y)] = npmi
	return outdict

#c, e, m = get_label_combis(sys.argv[1])
c, e, m = get_measures_emotions(sys.argv[1])
c = get_probs(Counter(c))
e = get_probs(Counter(e))
m = get_probs(Counter(m))
#print(c)
#print(len(c))
#sleep(2)
#print(e)
#print(len(e))
#sleep(2)
#print(m)
#print(len(m))
#sleep(2)
final_dict = calc_npmi(c, m, e)
#print(final_dict)
sorted_npmi = {k: v for k, v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
#print(sorted_npmi)
measures = {}
emotions = {}
for i, n in sorted_npmi.items():
	m, e = i
	#if e == 'Suspense':
	#	print(m, '\t',  n)
	measures[m] = 0
	emotions[e] = 0

matrix = []
measures_labels = sorted(measures.keys())
emotions_labels = emotions.keys()
for m in emotions_labels:
	row = []
	for e in measures_labels:
		row.append(round(sorted_npmi[(e,m)], 2))
	matrix.append(row)

matrix = np.array(matrix)
print(matrix)
print(measures_labels)
print(emotions_labels)



#measures, emotions = zip(*sorted_npmi.keys())


#emotions_labels = ["cucumber", "tomato", "lettuce", "asparagus",
#              "potato", "wheat", "barley"]
#measures_label = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

#matrix = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots(1,1,figsize=(10,8))
plotpath = 'plots/npmi/heatmap_measures_emotions_short_nogreek.png'	
#cmap = matplotlib.cm.Pastel1_r
#cmap = matplotlib.cm.RdYlBu
#cmap = matplotlib.cm.jet
#cmap = matplotlib.cm.cool
#cmap_r = ListedColormap(cmap.colors[::-1])
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)


cmap = matplotlib.colors.ListedColormap(['black', 'green', 'teal', 'turquoise', 'honeydew', #'lavender',
                                  'lemonchiffon', 'orange', 'tomato', 'crimson', 'hotpink'])
cmap.set_over('violet')
cmap.set_under('black')

bounds = [-1, -0.99, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 0.99, 1]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
cb3 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                boundaries= bounds, #[-0.9] + bounds + [0.9],
                                extend='both',
                                extendfrac='auto',
                                ticks=bounds,
                                spacing='uniform',
                                orientation='vertical')
cb3.set_label('NPMI')

#fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cb3, norm=norm), cax=cax)
#fig.colorbar(cb3, cax=cax)

# We want to show all ticks...
ax.set_xticks(np.arange(len(measures_labels)))
ax.set_yticks(np.arange(len(emotions_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(measures_labels, fontsize=16)
ax.set_yticklabels(emotions_labels, fontsize=18)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(emotions_labels)):
    for j in range(len(measures_labels)):
        text = ax.text(j, i, matrix[i, j],
                       ha="center", va="center", color="black", fontsize=12)


#fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
im = ax.imshow((matrix), vmin=-1, vmax=1, cmap=cmap, norm=norm)

ax.set_title("Verse Measures vs. Aesthetic Emotions: NPMI", fontsize=24)
fig.tight_layout()
#plt.show()
fname = plotpath #'Measures_and_Emotions_imeasures_nogreek_rdgy.png'	
plt.savefig(fname)
