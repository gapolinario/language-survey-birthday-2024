import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode

def print_tally_by_key(dict):

    for key, value in sorted(dict.items()):
        print(f"{key} : {value}")

def print_tally_by_value(dict,type):

    for key, value in sorted(dict.items(), key=lambda item: -item[1]):
        if type == 'i':
            print(f"{key} : {value}")
        elif type == 'f':
            print(f"{key} : {value:.2f}")
        else:
            raise ValueError(f"Type must be i or f, received {type}")

# https://stackoverflow.com/a/73905572
def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

def weighted_mode(values, weights):
    return values[np.argmax(weights)]

langs = ['L1','L2','L3','L4','L5','L6','L7','L8','L9']
cols = ['Name']+langs
df = pd.read_csv('survey.csv', sep=',', header=None, names=cols)
#print(df.values)

## PART 1: Tallies

# Tally of L1 languages
print('')
print(df.value_counts('L1'))

# Tally of L2 languages
print('')
print(df.value_counts('L2'))

# Tally of L3 languages
print('')
print(df.value_counts('L3'))

# Histogram: Number of languages per person
# Sorted in ascending order by number of languages
langs_per_person = df.count(axis=1).to_numpy()
unique_langcount, count_langcount = np.unique(langs_per_person, return_counts=True)
unique_langcount -= 1

langs_per_person_dict = dict(zip(unique_langcount, count_langcount))

print('\nNumber of languages : Number of speakers')
print_tally_by_key(langs_per_person_dict)

print('')
print('Number of languages per person:')
m1 = np.average(unique_langcount,weights=count_langcount)
m2 = np.sqrt( np.average(unique_langcount**2,weights=count_langcount) - m1**2 )
md = weighted_median(unique_langcount,weights=count_langcount)
mo = weighted_mode(unique_langcount,weights=count_langcount)
print(f'Mean: {m1:.1f}')
print(f'Mode: {mo:.1f}')
print(f'Median: {md:.1f}')
print(f'Stdev: {m2:.1f}')

# Histogram: Number of speakers of each language
# Sorted in descending order by number of speakers
all_langs = df[langs].to_numpy().flatten()
all_langs = [i for i in all_langs if i is not np.nan]

unique_langs, count_langs = np.unique(all_langs, return_counts=True)
all_langs_dict = dict(zip(unique_langs, count_langs))

print('\nLanguage : Number of speakers')
print_tally_by_value(all_langs_dict,'i')

#print('')
#print(unique_langs)

## PART 2: Predictive

# Threshold for predicting number of languages
thresh = 4
thresh_index = cols[thresh]

# Probability that a person speaks geq threshold languages,
# for each language
prob_cond = {}

for lang, num_speakers in all_langs_dict.items():

    condition = num_speakers > 1
    #condition = True

    if condition:

        # number of speakers of language lang that speak more than thresh languages
        gt_thresh = 0

        # for each speaker, if they speak the target language
        # and speak more than the threshold number, add 1 to the count
        for i, row in df.iterrows():
            if (row == lang).any() and (not row.isna()[thresh_index]):
                gt_thresh += 1

        prob_cond[lang] = gt_thresh / num_speakers

print(f'\nLanguage : Probability that speaker speaks {thresh}+ languages')
print_tally_by_value(prob_cond,'f')
