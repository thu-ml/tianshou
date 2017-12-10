from rank_based import *
from proportional import *
from naive import *
import sys

def getReplayBuffer(name, conf):
	'''
	Get replay buffer according to the given name.
	'''
	if (name == 'rank_based'):
		return RankBasedExperience(conf)
	elif (name == 'proportional'):
		return PropotionalExperience(conf)
	elif (name == 'naive'):
		return NaiveExperience(conf)
	else:
		sys.stderr.write('no such replay buffer')
