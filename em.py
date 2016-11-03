import csv
import math
import random
import numpy as np
import scipy.stats as stats

#A matrix where each row is ten flips of one of the selected coins
observations = []
#The number of coins used in the experiment
num_coins = 3
#The probability of flipping a heads for each coin
heads_prob = [0.15, 0.4, 0.8]
#The probability of selecting the coin
pick_prob = [0.2, 0.5, 0.3]
#The number of folds for k-cross fold validation
K = 10

def em_single(priors, observations, num_coins):
    """Performs a single EM step
    Args:
        priors (int[]): An array of the probability of the corresponding coin landing on heads i.e. theta_A, theta_B
        observations (str[][]): The file we read in of  experiment observations
    
    Returns:
        new_priors: [new_theta_A, new_theta_B]
    """
    #counts is a dictionary of coin number to a Heads and Tails count
    counts = {}
    for x in xrange(num_coins):
        counts[x] = {'H':0,'T':0}
    # E step
    for o in observations: 
        len_observation = len(o)
        num_heads = sum(map(lambda x: 1 if x == 'H' else 0, o))
        num_tails = len_observation - num_heads
        #Find Probability Mass Function (PMF)
        contributions = []
        for x in xrange(num_coins):
            contributions.append(stats.binom.pmf(num_heads,len_observation, priors[x]))
        #Expectation Maximization
        weights = []
        for x in xrange(num_coins):
            weights.append(contributions[x]/sum(contributions))
        # Incrementing counts
        i = 0
        for x in counts:
            counts[x]['H']+=weights[i]*num_heads
            counts[x]['T']+=weights[i]*num_tails
            i+=1
    # M step
    new_pick_probs = []
    for x in counts:
        new_pick_probs.append(round(counts[x]['H']/(counts[x]['H'] + counts[x]['T']), 3))
    return new_pick_probs

def em(observations, prior, num_coins):
    """ 
    Drives the iterations of EM. Does either 10000 iterations of 
    learning or goes until we get less than 1e-4 accuracy improvement from iterating again.

    Args:
        observations (str[]): The observations we read in from the experiment
        prior (): The prior P(H | Ci) estimate
        num_coins (int): The number of coins used in the experiment
    Returns: 
        An array of P(H | Ci) and the number of iterations it took to get there.

    """
    max_iterations = 10000
    tol = 1e-4
    i = 0

    while i < max_iterations:
        new_prior = em_single(prior, observations, num_coins)
        delta_change = np.abs(prior[0]-new_prior[0])
        if delta_change > tol:
            prior = new_prior
            i+=1
        else:
            break
    return [new_prior, i]

def make_probs(divs):
    """
    Generates a list with the specified number of divisions, for initial "guesses" of prior values.

    Args:
        divs (int): the number of divisiona to have
    Returns:
        A list of random probabilities that sum to 1
    """
    if divs == 1:
        return [0.99]
    probs = []
    probs.append(random.random())
    for i in xrange(divs-1):
        if i == divs-2:
            probs.append(1 - sum(probs))
        else:
            probs.append(random.uniform(0, 1-sum(probs)))
    return probs

def f1(precision, recall):
    """ This function allows us to have a value to select for when finding 
        the ideal number of clusters. (https://en.wikipedia.org/wiki/F1_score)

    Args:
        precison (float): The number of correct positive results
            divided by the number of all positive results.
        recall (float): The number of correct positive results divided by the
            number of positive results that should have been returned.

    Returns:
        A weighted value for determining the effectiveness of the results form clustering.
    """
    return 2 * (1/((1/recall) + (1/precision)))

def cross_fold(k, data):
    """Breaks the data array into k folds.

    Args:
        k (int): The number of folds to divide teh data into.
        data (char[]): The raw heads or tails obervation data to be folded.

    Returns:
        The given data as a matrix where each row is an evenly distributed 
        subsection of the data.
    """
    #An array to hold k rows of even sized data
    folds = [[] for r in xrange(k)]
    i = 0
    for d in data:
        folds[i].append(d)
        i = i+1 if i < k-1 else 0
    return folds

# Open our file and start reading it 
with open("eggs.csv") as file:
    reader = csv.reader(file)
    for line in reader:
        observations.append(line)

#Run EM with 1-10 coins (clusters)
cross_folds = cross_fold(K, observations)
f1_values = [[] for r in xrange(10)]
for n in xrange(1, 11):
    for i in xrange(K):
        test_fold = cross_folds.pop(i) #Separate test fold & training set
        p = random.random()
        r = random.random()
        f1_values[n-1].append(f1(p, r))
        cross_folds.insert(i, test_fold) #Put back test fold
    f1_values[n-1] = sum(f1_values[n-1])/len(f1_values[n-1])
    # print em(observations, make_probs(n),  n)
print f1_values

# print "Number of coins = %d" % num_coins
# for i in range(len(heads_prob)):
#     print "P(H|c%d) = %.2f" % (i+1, heads_prob[i]) 