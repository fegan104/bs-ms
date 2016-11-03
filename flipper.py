import csv
import random

def do_ten_flips(num_coins, heads_prob):
	with open('eggs.csv', 'a') as csvfile:
		flipwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		observations = []
		for f in xrange(10):
			if random.random() < heads_prob:
				observations.append('H')
			else:
				observations.append('T')
		flipwriter.writerow(observations)

def do_experiment():
	#A matrix where each row is ten flips of one of the selected coins
	observations = []
	#The number of coins used in the experiment
	num_coins = 5
	#The probability of flipping a heads for each coin
	heads_prob = [0.15, 0.44, 0.80, 0.23]
				#[0.22, 0.55, 0.81, 0.31]
	#The probability of selecting the coin
	pick_prob = [0.10, 0.4, 0.3, 0.20]
	#
	divs = []
	for x in xrange(len(pick_prob)+1):
		temp = 0
		for y in xrange(x):
			temp += pick_prob[y]
		divs.append(temp)

	for x in xrange(836):
		point = random.random()
		for x in xrange(1, len(divs)):
			if divs[x-1] < point < divs[x]:
				do_ten_flips(num_coins, heads_prob[x-1])


do_experiment()

