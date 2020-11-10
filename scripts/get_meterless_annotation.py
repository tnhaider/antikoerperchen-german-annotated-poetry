import sys, re
from inout.dta.corpus import Corpus
from inout.dta.document import Document
from inout.dta.poem import Poem
from inout.utils.helper import * 

c = Corpus(sys.argv[1])
poems = c.read_poems()

for poem in poems:
	print(poem.get_title())
	for stanza in poem.get_stanzas():
		print()
		for line in stanza.get_line_objects():
			text = line.get_text()
			footmeter = line.get_meter()
			if not footmeter:
				continue
			meter = re.sub('\|', '', footmeter)
			measure = get_versification(meter)
			smeasure = measure.split('.')[0]
			print(meter, '\t', measure, '\t', smeasure, '\t', text, '\t', footmeter)
