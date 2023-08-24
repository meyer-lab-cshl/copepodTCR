#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from itertools import combinations
import cvxpy as cp
import random
from collections import Counter

def string_overlap(str1, str2):
    
    """
    Takes two peptides, returns length of their overlap.
    """
    
    overlap_len = 0
    for i in range(1, min(len(str1), len(str2)) + 1):
        if str1[-i:] == str2[:i]:
            overlap_len = i
    return overlap_len

def all_overlaps(strings):
    
    """
    Takes list of peptides, returns occurence of overlap of different lengths.
    """
    
    overlaps = []
    for i in range(len(strings) - 1):
        overlaps.append(string_overlap(strings[i], strings[i+1]))

    return Counter(overlaps)

def find_pair_with_overlap(strings, target_overlap):
    
    """
    Takes list of peptides and overlap length.
    Returns peptides with this overlap.
    """
    
    target = []
    for i in range(len(strings) - 1):  
        if string_overlap(strings[i], strings[i+1]) == target_overlap:
            target.append([strings[i], strings[i+1]])
    return target