#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from itertools import combinations
import cvxpy as cp
import random
from collections import Counter


# # Gray codes functions

# In[17]:


def find_q_r(n):
    
    """
    Solves an equation: what is an equal for partition for 2**n:
    2**n = n*q + r
    What is n?
    Used in function(bgc).
    """

    q = cp.Variable(integer=True)
    r = cp.Variable(integer=True)

    constraints = [
        2**n == n*q + r,
        r >= 0,
        r <= n-1
    ]

    problem = cp.Problem(cp.Minimize(r), constraints)

    problem.solve()
    
    if problem.status == 'optimal':
        return int(q.value), int(r.value)
    
def bgc(n, s = None):
    
    """
    Balanced Gray codes construction.
    Takes a transition sequence for a balanced Gray code with n-2 bits,
    returns a transition sequence of n-bit BGC.
    Is dependent on function(find_q_r).
    Used in function(n_bgc)
    """

    ### Calculation of q, r
    q, r = find_q_r(n=n)

    ### Partition p_i
    p_i = []

    if q%2 == 0:
        q_def = int(r/2)
        if q_def != 0:
            Q = list(range(1, n-1))[-q_def:]
        else:
            Q = []
    
        for i in range(n):
            if i in Q:
                p_i.append(q+2)
            else:
                p_i.append(q)
    elif q%2 != 0:
        q_def = int((n+r)/2)
        if q_def != 0:
            Q = list(range(1, n-1))[-q_def:]
        else:
            Q = []
    
        for i in range(n):
            if i in Q:
                p_i.append(q+1)
            else:
                p_i.append(q-1)
            
    p_i = sorted(p_i)

    ### Calculation b_i
    if s is None:
        if n == 4:
            s = [1, 2, 1, 2]
        elif n == 5:
            s = [1, 2, 3, 2, 1, 2, 3, 2]
    b_i = []

    for i in range(1, len(set(s))+1):
        if i != s[len(s)-1]:
            b = (4*s.count(i) - p_i[i-1])/2
            b_i.append(int(b))
        else:
            b = (4*(s.count(i) - 1) - p_i[i-1])/2
            b_i.append(int(b))
    l = sum(b_i)

    counts = dict()
    for i in range(len(b_i)):
        counts[i+1] = b_i[i]
    
    s = s[:-1]
    u = []
    t = []
    new_counts = dict()
    for i in range(1, n-1):
        new_counts[i] = 0
    for i in s:
        if new_counts[i] >= counts[i]:
            u[-1].append(i)
        else:
            t.append([i])
            u.append([])
        new_counts[i] += 1
    n = n-2

    s_2 = []

    for t_i, u_i in zip(t, u):
        s_2 = s_2 + t_i + u_i
    s_2 = s_2 + [n+1]

    row_count = 0
    for i in range(len(u)-1, -1, -1):
        if row_count == 0:
            s_2 = s_2 + list(reversed(u[i])) + [n+2] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
            row_count = 1
        else:
            s_2 = s_2 + list(reversed(u[i])) + [n+1] + u[i] + [n+2] + list(reversed(u[i])) + t[i]
            row_count = 0
    if row_count == 0:
        s_2 = s_2 + [n+2] + [n+1] + [n+2]
    elif row_count == 1:
        s_2 = s_2 + [n+1] + [n+2] + [n+1]

    return s_2

def n_bgc(n):
    
    """
    Takes n and returns n-bit BGC.
    Is dependent on function(bgc).
    Used in function(m_length_BGC).
    """
    
    if n == 2:
        s_2 = [1, 2, 1, 2]
        counter = 2
    elif n == 3:
        s_2 = [1, 2, 3, 2, 1, 2, 3, 2]
        counter = 3
    elif n >3 and n%2 == 0:
        counter = 4
        s_2 = bgc(n=counter)
    elif n > 3 and n%2 != 0:
        counter = 5
        s_2 = bgc(n=counter)
    while counter != n:
        counter = counter + 2
        s_2 = bgc(n=counter, s = s_2)
        
    balance = []
    for item in set(s_2):
        balance.append(s_2.count(item))
        
    #print(balance)
    return s_2

def computing_ab_i_odd(s_2, l, v):
    
    """
    Used in special case of n-bit BGC construction with flexible length.
    Used in function(m_length_BGC).
    """
    
    ## How many values we need to add before s_r
    E_v = int(np.floor((v-1)/3))
    E_v = s_2[:E_v]
        
    ## Computing b_i
    b_i = dict()
    for i in range(n):
        b_i[i] = 0
        if i in E_v:
            b_i[i] = E_v.count(i)
            
    inequalities = []
    TC = dict()

    ## How many a_i we need to compute:
    a_i = []
    for i in range(n):
        a_i.append(cp.Variable(integer=True))

    for i in range(n+2):
        if l%2 == 0:
            if i == n:
                TC_i = l - cp.floor(v/3) + cp.ceil(((v+4)%6)/6)
            elif i == n+1:
                TC_i = l - cp.floor(v/3) + cp.ceil(((v+1)%6)/6)
            elif i == s_2[-1]:
                TC_i = 3*(s_2.count(i)-1) - 2*a_i[i] + b_i[i]
            else:
                TC_i = 3*s_2.count(i) - 2*a_i[i] + b_i[i]
            TC[i] = TC_i
        else:
            if i == n:
                TC_i = l - cp.floor(v/3) + cp.ceil(((v+1)%6)/6)
            elif i == n+1:
                TC_i = l - cp.floor(v/3) + cp.ceil(((v+4)%6)/6)
            elif i == s_2[-1]:
                TC_i = 3*(s_2.count(i)-1) - 2*a_i[i] + b_i[i]
            else:
                TC_i = 3*s_2.count(i) - 2*a_i[i] + b_i[i]
            TC[i] = TC_i
                
    ## Solving the resulting inequalities for a_i
    inequalities = []
    for key1 in TC.keys():
        for key2 in TC.keys():
            if key1 != key2:
                inequalities.append(-2 <= TC[key1] - TC[key2])
                inequalities.append(TC[key1] - TC[key2] <= 2)
    inequalities.append(sum(a_i) == l)
    for i in range(len(a_i)):
        inequalities.append(a_i[i] >= 0)
        inequalities.append(a_i[i] <= l)

    a_values = dict()
    problem = cp.Problem(cp.Minimize(0), inequalities)
    problem.solve()

    if problem.status == 'optimal':
        for i in range(len(a_i)):
            a_values[i] = int(a_i[i].value)
    
    return [v, a_values, E_v]

### Ready for both cases
def m_length_BGC(m, n):
    
    """
    Construction of n-bit BGC with flexible length from n-2 bit BGC.
    Is dependent on function(computing_ab_i_odd) and function(n_bgc).
    """
    
    n = n-2
    s_2 = n_bgc(n = n)
    s_2 = [x - 1 for x in s_2]
    
    ### if 3*2**n < m < 2**(n+2) (Case I)
    if 3*2**n < m < 2**(n+2):
        intervals = [np.floor(m/(n+2)) -3, np.floor(m/(n+2))]
    
        ## l is chosen from intervals
        l_options = dict()
        for l in list(range(int(intervals[0]), int(intervals[1]) + 1)):
            ## How many values we need to add before s_r
            u = m - 3*2**n
            if l%2 == 0:
                E_u = s_2[-l:][:-1]
            elif l%2 != 0:
                E_u = s_2[-l-1:][:-1]
        
            ## Computing b_i
            b_i = dict()
            for i in range(n):
                b_i[i] = 0
                if i in E_u:
                    b_i[i] = E_u.count(i)

            inequalities = []
            TC = dict()

            ## How many a_i we need to compute:
            a_i = []
            for i in range(n):
                a_i.append(cp.Variable(integer=True))

            for i in range(n+2):
                if l%2 == 0:
                    if i == n:
                        TC_i = l + 2
                    elif i == n+1:
                        TC_i = l + 2
                    elif i == s_2[-1]:
                        TC_i = 3*(s_2.count(i)-1) - 2*a_i[i] + b_i[i]
                    else:
                        TC_i = 3*s_2.count(i) - 2*a_i[i] + b_i[i]
                    TC[i] = TC_i
                else:
                    if i == n:
                        TC_i = l + 2
                    elif i == n+1:
                        TC_i = l + 1
                    elif i == s_2[-1]:
                        TC_i = 3*(s_2.count(i)-1) - 2*a_i[i] + b_i[i]
                    else:
                        TC_i = 3*s_2.count(i) - 2*a_i[i] + b_i[i]
                    TC[i] = TC_i
                
            ## Solving the resulting inequalities for a_i
            inequalities = []
            for key1 in TC.keys():
                for key2 in TC.keys():
                    if key1 != key2:
                        inequalities.append(-2 <= TC[key1] - TC[key2])
                        inequalities.append(TC[key1] - TC[key2] <= 2)
            for i in range(len(a_i)):
                inequalities.append(a_i[i] >= 0)
                inequalities.append(a_i[i] <= l)
            inequalities.append(sum(a_i) == l)

            a_values = dict()
            problem = cp.Problem(cp.Minimize(0), inequalities)
            problem.solve()

            if problem.status == 'optimal':
                for i in range(len(a_i)):
                    a_values[i] = int(a_i[i].value)
                break
            l_options[l] = [u, a_values]
                    
        s_2 = s_2[:-1]
        u = []
        t = []
        new_counts = dict()
        for i in range(0, n):
            new_counts[i] = 0
        for i in s_2:
            if new_counts[i] >= a_values[i]:
                u[-1].append(i)
            else:
                t.append([i])
                u.append([])
            new_counts[i] += 1
    
        flex_s = []
        if l%2 == 0:
            flex_s = flex_s + E_u + [n]
            row_count = 0
            for i in range(l-1, -1, -1):
                if row_count == 0:
                    flex_s = flex_s + list(reversed(u[i])) + [n+1] + u[i] + [n] + list(reversed(u[i])) + t[i]
                    row_count = 1
                elif row_count == 1:
                    flex_s = flex_s + list(reversed(u[i])) + [n] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
                    row_count = 0
            flex_s = flex_s + [n+1] + [n] + [n+1]
    
        elif l%2 != 0:
            flex_s = flex_s + E_u + [n]
            row_count = 0
            for i in range(l-1, -1, -1):
                if row_count == 0:
                    flex_s = flex_s + list(reversed(u[i])) + [n+1] + u[i] + [n] + list(reversed(u[i])) + t[i]
                    row_count = 1
                elif row_count == 1:
                    flex_s = flex_s + list(reversed(u[i])) + [n] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
                    row_count = 0
            flex_s = flex_s + [n] + [n+1] + [n]
    
            
        balance = []
        for item in set(flex_s):
            balance.append(flex_s.count(item))
        #print(balance)
    
        return flex_s
    
    ### if 2**(n+1) < m <= 3*(2**n) (Case II)
    if 2**(n+1) < m <= 3*(2**n):
        v = 3*(2**n)-m
        intervals = [np.floor(m/(n+2)) + np.floor(v/3) -2, np.floor(m/(n+2)) + np.floor(v/3) +2]
    
        ## Possible l's and v's:
        l_options = dict()
    
        ## l is chosen from intervals
        for l in list(range(int(intervals[0]), int(intervals[1]) + 1)):
            l_options[l] = computing_ab_i_odd(s_2 = s_2, l = l, v = v)
            
            if l_options[l][1] != {}:
                v = l_options[l][0]
                if v > 1:
                    el = int(np.floor((v+1)/3))
                    t = s_2[:el]
                    a_i = l_options[l][1]
                    verdict = []
                    for item in a_i.keys():
                        if a_i[item] != t.count(item):
                            verdict.append('No')
                        else:
                            verdict.append('True')
                    if a_i == {}:
                        verdict.append('No')
                        
                    if 'No' not in verdict:
                        u = []
                        t = []
                        new_counts = dict()
                        for i in range(0, n):
                            new_counts[i] = 0
                        for i in s_2:
                            if new_counts[i] >= a_values[i]:
                                u[-1].append(i)
                            else:
                                t.append([i])
                                u.append([])
                            new_counts[i] += 1
                        
                        flex_s = []
                        if l%2 == 0:
                            row_count = 0
                            for i in range(l-1, -1, -1):
                                if row_count == 0:
                                    flex_s = flex_s + list(reversed(u[i])) + [n+1] + u[i] + [n] + list(reversed(u[i])) + t[i]
                                    row_count = 1
                                elif row_count == 1:
                                    flex_s = flex_s + list(reversed(u[i])) + [n] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
                                    row_count = 0
                                    flex_s = flex_s + [n+1] + [n] + [n+1]
    
                        elif l%2 != 0:
                            row_count = 0
                            for i in range(l-1, -1, -1):
                                if row_count == 0:
                                    flex_s = flex_s + list(reversed(u[i])) + [n+1] + u[i] + [n] + list(reversed(u[i])) + t[i]
                                    row_count = 1
                                elif row_count == 1:
                                    flex_s = flex_s + list(reversed(u[i])) + [n] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
                                    row_count = 0
                                    flex_s = flex_s + [n] + [n+1] + [n]
                        flex_s = flex_s[:-v] 
                        balance = []
                        for item in set(flex_s):
                            balance.append(flex_s.count(item))
                        #print(balance)
                        return flex_s
                    
                    elif 'No' in verdict:
                        new_options = dict()
                        new_s = s_2[1:] + [s_2[0]]
                        new_options[l] = computing_ab_i_odd(s_2 = new_s, l = l, v = v)
                        v = new_options[l][0]
                        if v > 1:
                            el = int(np.floor((v+1)/3))
                            t = new_s[:el]
                            a_i = new_options[l][1]
                            verdict = []
                            for item in a_i.keys():
                                if a_i[item] != t.count(item):
                                    verdict.append('No')
                                else:
                                    verdict.append('True')
                            if a_i == {}:
                                verdict.append('No')
                        
                            if 'No' not in verdict:
                                
                                u = []
                                t = []
                                new_counts = dict()
                                for i in range(0, n):
                                    new_counts[i] = 0
                                for i in s_2:
                                    if new_counts[i] >= a_values[i]:
                                        u[-1].append(i)
                                    else:
                                        t.append([i])
                                        u.append([])
                                    new_counts[i] += 1
                                
                                flex_s = []
                                if l%2 == 0:
                                    row_count = 0
                                    for i in range(l-1, -1, -1):
                                        if row_count == 0:
                                            flex_s = flex_s + list(reversed(u[i])) + [n+1] + u[i] + [n] + list(reversed(u[i])) + t[i]
                                            row_count = 1
                                        elif row_count == 1:
                                            flex_s = flex_s + list(reversed(u[i])) + [n] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
                                            row_count = 0
                                            flex_s = flex_s + [n+1] + [n] + [n+1]
    
                                elif l%2 != 0:
                                    row_count = 0
                                    for i in range(l-1, -1, -1):
                                        if row_count == 0:
                                            flex_s = flex_s + list(reversed(u[i])) + [n+1] + u[i] + [n] + list(reversed(u[i])) + t[i]
                                            row_count = 1
                                        elif row_count == 1:
                                            flex_s = flex_s + list(reversed(u[i])) + [n] + u[i] + [n+1] + list(reversed(u[i])) + t[i]
                                            row_count = 0
                                            flex_s = flex_s + [n] + [n+1] + [n]
                                flex_s = flex_s[:-v]
                                balance = []
                                for item in set(flex_s):
                                    balance.append(flex_s.count(item))
                                #print(balance)
                                return flex_s
                            
                            
def gc_to_address(s_2, iters, n):
    
    """
    Takes BGC transition sequence and returns BGC with particular number of 1 (iters).
    Returns list of addresses.
    """
    
    codes = [['0']*n]
    for item in s_2:
        n_item = codes[-1].copy()
        if n_item[item-1] == '0':
            n_item[item-1] = '1'
        else:
            n_item[item-1] = '0'
        codes.append(n_item)
    addresses = []
    for item in codes:
        if item.count('1') == iters:
            ad = []
            for i in range(len(item)):
                if item[i] == '1':
                    ad.append(i)
            if ad not in addresses:
                addresses.append(ad)
    return addresses