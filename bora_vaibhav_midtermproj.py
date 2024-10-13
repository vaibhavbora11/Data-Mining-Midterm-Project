#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Importing all the necessary libraries
import pandas as pd
import itertools
import time
import os
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Welcome
print("Welcome!!!")
datasets = ('Amazon', 'Best Buy', 'K-Mart', 'Nike', 'Supermarket') # These are all 5 of my datasets
store_selection = input("Select which store you want to choose: \n1. Amazon\n2. Best Buy\n3. K-Mart\n4. Nike\n5. Supermarket\n6. Exit\n")
user_min_supp = int(input("Enter the minimum support in % value between 1-100: "))
user_min_con = int(input("Enter the minimum confidence in % value between 1-100: "))

# Quit if the user presses 6
if store_selection == '6':
    print("Goodbye!!!")
    quit()

# Incorrect input if user inputs an invalid answer
try:
    selected_index = int(store_selection) - 1
    if selected_index not in range(len(datasets)):
        raise ValueError("The input is not within range")
except ValueError as e:
    print("Error. Please enter a digit between 1-5.") # Message will say that the input should be between 1-5
    quit()

# Load datasets according to user input
store_name = datasets[selected_index]
transactions_file = f"{store_name} Transactions.csv"
item_names_file = f"{store_name} Item Names.csv"

# Reading the files (make sure they are in the same directory)
try:
    transactions_df = pd.read_csv(transactions_file, encoding='ISO-8859-1') # Had an error reading my csv files because it did not have default UTF-8 characters
    item_names_df = pd.read_csv(item_names_file, encoding='ISO-8859-1')
    print(f"Nice choice! You have selected {store_name}.")
except FileNotFoundError:
    print(f"Error was detected trying to access the {store_name} files.")
except Exception as e:
    print(f"A weird error happened: {e}")

#----------------------------------------------------------------------------------------------------------------------------

# 1-FREQUENT ITEMSET
start1 = time.time() # using time to calculate the execution time of each function
transactions_df['Items'] = transactions_df['Transactions'].str.split(', ')
transactions_list = transactions_df['Items'].tolist()

def one_freq_itemset(transactions_list): # Function one_freq_itemset will be calculating the frequency of each item in all transactions
    one_freq = {}
    for transaction in transactions_list: # Using for loop
        for item in transaction:
            if item in one_freq:
                one_freq[item] += 1 # Increment
            else:
                one_freq[item] = 1 # Keep the same
    return one_freq

one_freq = one_freq_itemset(transactions_list) # Using one_freq to call the function
one_freq_df = pd.DataFrame(list(one_freq.items()), columns=['Itemset', 'Support']) # Converting to a dataframe

total_transactions = len(transactions_list) # Using len() to determine the total number of items
one_freq_df['Support'] = one_freq_df['Support'] / total_transactions # Calculating support of each item
min_support_count = (user_min_supp / 100) * total_transactions # Calculating minimum support count
min_support_threshold = user_min_supp / 100 # Calculating minimum support threshold

frequent_itemsets_one_freq = one_freq_df[one_freq_df['Support'] >= min_support_threshold] # Making it so that the dataframe will only include items which have support more than the minimum
print("\n1-Frequent Itemsets from Apriori using hardcode:")
print(frequent_itemsets_one_freq) # Output
end1 = time.time()
print("\nThe time of execution of above program is :", (end1-start1) * 10**3, "ms")

#----------------------------------------------------------------------------------------------------------------------------

# 2-FREQUENT ITEMSET
start2 = time.time()
def all_2_combos(prev_frequent_items): # Made a function to create a list of all the possible combinations
    return list(itertools.combinations(prev_frequent_items, 2))

frequent_items_1 = frequent_itemsets_one_freq['Itemset'].tolist() # Calling all the frequent items from the prev calculation
pair_2 = all_2_combos(frequent_items_1) # Possible pairs

def pair_2_support(candidates, transactions_list): # This function will calculate support of each possible pair
    goodpair_2 = {} # This will be the list of all the support count
    for candidate in candidates: # Using for loop
        candidate_set = set(candidate)
        goodpair_2[candidate] = sum(1 for transaction in transactions_list if candidate_set.issubset(set(transaction)))
    return goodpair_2

goodpair_2items = pair_2_support(pair_2, transactions_list) # Finding the support
pair_2_df = pd.DataFrame(list(goodpair_2items.items()), columns=['Itemset', 'Support']) # Converting to dataframe
pair_2_df['Support'] = pair_2_df['Support'] / total_transactions
frequent_itemsets_two_freq = pair_2_df[pair_2_df['Support'] >= min_support_threshold] # Filtering the pair with only support more than minimum

print("\n2-Frequent Itemsets from Apriori using hardcode:")
print(frequent_itemsets_two_freq) # Output
end2 = time.time()
print("\nThe time of execution of above program is :", (end2-start2) * 10**3, "ms")

#----------------------------------------------------------------------------------------------------------------------------

# 3-FREQUENT ITEMSET
start3 = time.time()
def all_3_combos(prev_frequent_items): # Made a function to create a list of all the possible combinations
    possible_3_itemsets = set() # Using a set will avoid having duplicates
    for first in prev_frequent_items:
        for second in prev_frequent_items:
            if first != second:
                combined = tuple(sorted(set(first).union(set(second))))
                if len(combined) == 3:
                    possible_3_itemsets.add(combined)
    return possible_3_itemsets

def pair_3_support(candidates, transactions_list): # Calculate support of each possibility
    goodpair_3 = {}
    for candidate in candidates:
        candidate_set = set(candidate)
        goodpair_3[candidate] = sum(1 for transaction in transactions_list if candidate_set.issubset(set(transaction))) # Checking subset
    return goodpair_3

temp = [tuple(item) for item in frequent_items_1] # This temp variable contains the list of all the tuples for processing
candidates_3_pair = all_3_combos(temp)
candidate_supports_3 = pair_3_support(candidates_3_pair, transactions_list) # Calculate support
candidates_3_pair_df = pd.DataFrame(list(candidate_supports_3.items()), columns=['Itemset', 'Support']) # Dataframe
candidates_3_pair_df['Support'] = candidates_3_pair_df['Support'].apply(lambda x: x / total_transactions)
frequent_itemsets_C3 = candidates_3_pair_df[candidates_3_pair_df['Support'] >= min_support_threshold] # Filter

print("\n3-Frequent Itemsets from Apriori using hardcode:")
print(frequent_itemsets_C3) # Output
end3 = time.time()
print("\nThe time of execution of above program is :", (end3-start3) * 10**3, "ms")

#----------------------------------------------------------------------------------------------------------------------------

# Association Rules
te = TransactionEncoder()
te_ary = te.fit_transform(transactions_list)
df = pd.DataFrame(te_ary, columns=te.columns_)

minimum_support = user_min_supp / 100  # This will convert percentage to fraction
frequent_itemsets = apriori(df, min_support=minimum_support, use_colnames=True)
minimum_confidence = user_min_con / 100  # This will convert percentage to fraction
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minimum_confidence)

if 'frequent_itemsets' not in locals():
    frequent_itemsets = apriori(df, min_support=minimum_support/total_transactions, use_colnames=True) # Finding apriori
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=user_min_con/100) # Finding the association rules
print("\nAssociation Rules are:")
print("-" * 60)  # Line for better visibility
for index, rule in rules.iterrows(): # Output
    antecedents = ', '.join(rule['antecedents'])
    consequents = ', '.join(rule['consequents'])
    print(f"Rule {index + 1}: {antecedents}, {consequents}.")
    print(f"     - Support: {rule['support']:.4f}")
    print(f"     - Confidence: {rule['confidence']:.4f}")
    print("-" * 60)  # Line for better visibility

#----------------------------------------------------------------------------------------------------------------------------

#VERIFYING WITH BUILT IN PYTHON PACKAGE
start4 = time.time()
minimum_support = user_min_supp/100
minimum_confidence = user_min_con/100
transaction_encoder = TransactionEncoder()
encoded_transactions = transaction_encoder.fit_transform(transactions_list)
encoded_transactions_df = pd.DataFrame(encoded_transactions, columns=transaction_encoder.columns_)
frequent_itemsets_apriori = apriori(encoded_transactions_df, min_support=minimum_support, use_colnames=True) # Apriori
frequent_itemsets_fpgrowth = fpgrowth(encoded_transactions_df, min_support=minimum_support, use_colnames=True) # FPGrowth
rules = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=minimum_confidence) # Association Rules

print("\nFrequent Itemsets from Apriori using in-built python package:")
print(frequent_itemsets_apriori) # Apriori Output
print("\nFrequent Itemsets from FP-Growth using in-built python package:")
print(frequent_itemsets_fpgrowth) # FPGrowth Output
print("\nGenerated Association Rules:") # Association Rules Output
for i, rule in enumerate(rules.itertuples(index=False), 1):
    print(f"Rule {i}: {rule.antecedents} -> {rule.consequents} (Conf: {rule.confidence:.2f}, Supp: {rule.support:.2f})")
end4 = time.time()
print("\nThe time of execution of above program is :", (end4-start4) * 10**3, "ms")


# 

# In[ ]:




