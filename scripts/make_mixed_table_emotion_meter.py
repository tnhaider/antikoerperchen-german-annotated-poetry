import sys, re
from inout.dta.corpus import Corpus
from inout.dta.document import Document
from inout.dta.poem import Poem
from inout.utils.helper import * 

def map_lines_emotions(tsvfile):
	outdict = {}
	outdictshort = {}
	f = open(tsvfile, 'r').readlines()
	c = 0
	for line in f:
		c += 1
		print(c, len(f), line)
		line = line.strip()
		text, a11, a12, a21, a22 = line.split('\t')
		text = normalize_characters(str(text.strip()))
		outdict[text] = [a11, a12, a21, a22]
		t_short = text[:20]
		outdictshort[t_short] = [a11, a12, a21, a22]
	return outdict, outdictshort


m, m_short = map_lines_emotions(sys.argv[1])


c = Corpus(sys.argv[2])
poems = c.read_poems()


outfile = open('antikoerperchen.lines.prosody.emotions.tsv', 'w')
header = '\t'.join(['poem_id', 'stanza_id', 'total_lines_in_stanza', 'total_stanzas_in_poem', 'id_stanza_in_poem', 'id_line_in_poem', 'id_line_in_stanza', 'author', 'pub_year','period', 'title', 'line_text', 'emotion1anno1', 'emotion2anno1', 'emotion1anno2', 'emotion2anno2', 'measure', 'i_measure', 's_measure', 'foot_meter', 'meter', 'feet', 'caesura_rhythm', 'main_accents', 'caesuras', 'rhyme_schema', 'enjambement', 'cadence', 'foot_inversion', 'chol_iambic', 'relaxed_syllables'])

outfile.write(header)
outfile.write('\n')

pix = 0
six = 0
pilix = 0
tab = '\t'
for poem in poems:
	pix += 1
	title = poem.get_title()
	author = poem.get_author()
	year = poem.get_year()
	stanzas = poem.get_stanzas()
	period = poem.get_period()
	silix = 0
	for stanza in stanzas:
		six += 1
		silix += 1
		print()
		lines = stanza.get_line_objects()
		rhyme = stanza.find_rhyme_schema()
		lix = 0
		for line in lines:
			pilix += 1
			lix += 1
			text = line.get_text()
			text = normalize_characters(text.strip())
			try:
				emotions = m[text]
			except KeyError:
			#	try:
			#		emotions = m_short[text[:20]]
			#	except KeyError:
				raise KeyError(text)
			if emotions[2] == 'NONE':
				emotions[2] = emotions[0]
			emotions = [re.sub(' / ', '/', i) for i in emotions]
			footmeter = line.get_meter()
			if not footmeter:
				continue
			meter = re.sub('\|', '', footmeter)
			measure = get_versification(meter)
			smeasure = measure.split('.')[0]
			imeasure = '.'.join(measure.split('.')[:2])
			caesurarhythm = line.get_rhythm()
			feet = ''.join(get_foot_anno(footmeter))
			caesura = ''.join(get_foot_anno(caesurarhythm))
			rhythm = re.sub('\|', '', caesurarhythm)
			caesurarhythm = re.sub('\|' , ':', caesurarhythm)
			enjambement = True
			if text[-1] in punct:
				enjambement = False
			invert = False
			if measure.split('.')[-1] == 'invert':
				invert = True
			chol = False
			if 'chol' in measure:
				chol = True
			cadence = 'male'
			if meter[-1] == '+':
				cadence = 'male'
			else: cadence = 'female'
			relaxed = False
			if 'relaxed' in measure:
				relaxed = True
			outlist = [str(pix), str(six), str(len(lines)), str(len(stanzas)), str(silix), str(pilix), str(lix), str(author), str(year), str(period), str(title), str(text), str(emotions[0]), str(emotions[1]), str(emotions[2]), str(emotions[3]), str(measure), str(imeasure), str(smeasure), str(footmeter), str(meter), str(feet), str(caesurarhythm), str(rhythm), str(caesura), str(rhyme), str(enjambement), str(cadence), str(invert), str(chol), str(relaxed)]
			#print(year c, '\t', meter, '\t', measure, '\t', smeasure, '\t', text, '\t', footmeter, '\t', emotions)
			#printlist = [i + tab for i in outlist]
			outfile.write('\t'.join(outlist) + '\n')
print(regex)
