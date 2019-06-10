import sys
import itertools

# (str)item_name : (int)count
one_itemsets_count = {}

# list of frequent one itemset's key
frequent_one_itemsets = []

# list of candidate two itemset's name (ex) "1 -> 2"
candidate_two_itemsets = []

# (str)itemset_name : (int) count
two_itemsets_count = {}

# (str)itemset_name : [(int)support, (int)confidence]
frequent_two_itemsets = {}

transaction_size = 0

# input parameters
datafile = open(sys.argv[1])
minsup = float(sys.argv[2])
minconf = float(sys.argv[3])

# Step 1: Find frequent 1-itemsets

# count 1-itemsets
for line in datafile:
    transaction_size += 1
    items = line.strip().split(',')
    items.pop(0)  # remove index value
    for item in items:
        if item in one_itemsets_count:
            one_itemsets_count[item] += 1
        else:
            one_itemsets_count[item] = 1

# find frequent 1-itemsets
for item in one_itemsets_count:
    if one_itemsets_count[item] / transaction_size >= minsup:
        frequent_one_itemsets.append(item)

# Step 2: Generate candidate 2-itemsets
candidate_two_itemsets = list(
    map(' -> '.join, itertools.permutations(frequent_one_itemsets, 2)))

# Step 3: Find frequent 2-itemsets

# count 2-itemsets
datafile.seek(0)
for line in datafile:
    items = line.strip().split(',')
    items.pop(0)  # remove index value
    # generate possible 2-itemsets in each line
    possible_two_itemsets = list(
        map(' -> '.join, itertools.permutations(items, 2)))
    # count
    for itemset in possible_two_itemsets:
        if itemset in two_itemsets_count:
            two_itemsets_count[itemset] += 1
        else:
            if itemset in candidate_two_itemsets:
                two_itemsets_count[itemset] = 1

# find frequent 2-itemsets
for item in two_itemsets_count:
    split_item = item.split(' -> ')
    support = two_itemsets_count[item] / transaction_size
    confidence = two_itemsets_count[item] / one_itemsets_count[split_item[0]]
    if support >= minsup and confidence >= minconf:
        frequent_two_itemsets[item] = [support, confidence]

# Step 4: Generate association rules

if len(frequent_two_itemsets) > 0:
    print("Association rules found:")
    for item in frequent_two_itemsets:
        print("{} (support = {}, confidence = {})".format(
            item, frequent_two_itemsets[item][0], frequent_two_itemsets[item][1]))
else:
    print("Association rule does not exist")

datafile.close()
