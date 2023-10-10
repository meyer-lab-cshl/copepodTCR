#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from itertools import combinations
import cvxpy as cp
import random
from collections import Counter
import trimesh
import sys
from io import StringIO
import zipfile
from io import BytesIO


# # Functions for ITERS search

def factorial(num):

    """
    Returns factorial of the number.
    Used in function(combination).
    """

    if num == 0:
        return 1
    else:
        return num * factorial(num-1)

def combination(n, k):

    """
    Returns number of possible combinations.
    Is dependent on function(factorial)
    Used in function(find_possible_k_values).
    """

    return factorial(n) // (factorial(k) * factorial(n - k))

def find_possible_k_values(n, l):

    """
    Returns possible iters given number of peptides (l) and number of pools (n).
    Is dependent on function(combination).
    """

    k_values = []
    k = 0
    
    while k <= n:
        c = combination(n, k)
        if c >= l:
            break
        k += 1

    while k <= n:
        if combination(n, k) >= l:
            k_values.append(k)
        else:
            break
        k += 1

    return k_values


# # Gray codes functions


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


# # Hamiltonian path functions


def union_address(address, union):
    
    """
    For AU-hamiltonian path search.
    Takes address and union, returns possible unions.
    Used in function(hamiltonian_path_AU).
    """
    
    one_bits = []
    zero_bits = []
    for i in range(len(address)):
        if address[i] == '1' and union[i] == '1':
            one_bits.append(i)
        elif address[i] == '0' and union[i] == '0':
            zero_bits.append(i)
    unions = []
    string = ['0']*len(union)
    for one_bit in one_bits:
        string[one_bit] = '1'
    for zero_bit in zero_bits:
        new_bit = string.copy()
        new_bit[zero_bit] = '1'
        unions.append(''.join(new_bit))
    return unions

def address_union(address, union):
    
    """
    For AU-hamiltonian path search.
    Takes union and address, returns possible addresses.
    Used in function(hamiltonian_path_AU).
    """
    
    one_bits = []
    for i in range(len(address)):
        if address[i] == '0' and union[i] == '1':
            zero_bit = i
        elif address[i] == '1' and union[i] == '1':
            one_bits.append(i)
    addresses = []
    string = ['0']*len(address)
    one_combs = list(combinations(one_bits, len(one_bits)-1))
    for one_comb in one_combs:
        new_bit = string.copy()
        new_bit[zero_bit] = '1'
        for one_bit in one_comb:
            new_bit[one_bit] = '1'
        addresses.append(''.join(new_bit))
    return addresses

def hamiltonian_path_AU(size, point, t, unions, path=None):
    
    """
    AU-hamiltonian path search.
    Is dependent on function(union_address), function(address_union), function(variance_score), function(sum_bits).
    Used in function(address_rearrangement_AU).
    """
    
    if path is None:
        path = []
    if unions is None:
        unions = []
    
    if t == 'a':
        if point not in set(path):
            path.append(point)
            if len(path) == size:
                return path
            next_points = union_address(address=path[-1], union=unions[-1] if unions else None)
            next_points.sort(key=lambda s: (variance_score(sum_bits(unions), s), random.random()))
            for nxt in next_points:
                res_path = hamiltonian_path_AU(size, nxt, 'u', unions, path)
                if res_path:
                    return res_path
            path.remove(point)
        else:
            return None
        
    elif t == 'u':
        if point not in set(unions):
            unions.append(point)
            next_points = address_union(address=path[-1], union=unions[-1])
            next_points.sort(key=lambda s: (variance_score(sum_bits(unions), s), random.random()))
            for nxt in next_points:
                res_path = hamiltonian_path_AU(size, nxt, 'a', unions, path)
                if res_path:
                    return res_path   
            unions.remove(point)
        else:
            return None
    return None

def variance_score(bit_sums, s):
    
    """
    For both versions of Hamiltonian path search.
    Takes an address (or union), measures how it influences the balance in path is being added.
    Returns penalty: difference between variance of balance before and after.
    Is dependent on function(bit_sums).
    Used in function(address_rearrangement_AU), function(address_rearrangement_A)
    """
    
    n = len(bit_sums)
    mean = sum(bit_sums) / n
    variance = sum((xi - mean) ** 2 for xi in bit_sums) / n

    new_bit_sums = bit_sums[:]
    for i, bit in enumerate(s):
        new_bit_sums[i] += int(bit)

    new_mean = sum(new_bit_sums) / n
    new_variance = sum((xi - new_mean) ** 2 for xi in new_bit_sums) / n

    penalty = new_variance - variance
    
    return penalty

def return_address_message(code, mode):
    
    """
    For A-hamiltonian path search.
    Takes an address and returns message (0/1 string).
    Or takes a message and returns an address.
    Used in function(binary_union).
    """
    
    if mode == 'a':
        address = []
        for i in range(len(code)):
            if code[i] == '1':
                address.append(i)
        return address
    if mode[0] == 'm':
        n = int(mode[1:])
        message = ''
        for i in range(n):
            if i in code:
                message = message + '1'
            else:
                message = message + '0'
        return message
    
def binary_union(bin_list):
    
    """
    For A-hamiltonian path search.
    Takes list of addresses, returns list of their unions.
    Is dependent on function(return_address_message).
    Used in function(hamiltonian_path_A).
    """
    
    union_list = []
    for i in range(len(bin_list)-1):
        
        set1 = set(return_address_message(bin_list[i], mode = 'a'))
        set2 = set(return_address_message(bin_list[i+1], mode = 'a'))
        set_union = set1.union(set2)
        union = return_address_message(set_union, mode = 'm'+str(len(bin_list[i])))
        union_list.append(union)
    
    return union_list

def hamming_distance(s1, s2):
    
    """
    For A-hamiltonian path search.
    Takes two messages (0/1 string) and returns their Hamming distance.
    Used in function(address_rearrangement_A).
    """
    
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def sum_bits(arr):
    
    """
    For both versions of hamiltonian path search.
    Takes list of addresses and returns their balance.
    Used in function(address_rearrangement_A), function(address_rearrangement_AU),
    function(hamiltonian_path_A), function(hamiltonian_path_AU).
    """
    
    bit_sums = [0]*len(arr[0])

    for s in arr:
        for i, bit in enumerate(s):
            bit_sums[i] += int(bit)
    return bit_sums

def hamiltonian_path_A(G, size, pt, path=None):
    
    """
    A-hamiltonian path search.
    Is dependent on function(binary_union), function(variance_score), function(sum_bits).
    Used in function(address_rearrangement_A).
    """

    if path is None:
        path = []
    if (pt not in set(path)) and (len(binary_union(path+[pt]))==len(set(binary_union(path+[pt])))):
        path.append(pt)
        if len(path)==size:
            return path
        next_points = G.get(pt, [])
        next_points.sort(key=lambda s: (variance_score(sum_bits(path), s), random.random()))
        for pt_next in next_points:
            res_path = hamiltonian_path_A(G, size, pt_next, path)
            if res_path:
                return res_path
        path.remove(pt)
    return None

def address_rearrangement_AU(n_pools, iters, len_lst):
    
    """
    For AU-hamiltonian path search.
    Takes number of pools, iters, and length of the path.
    Returns balance of the path and list of addresses.
    Is dependent on function(hamiltonian_path_AU) and function(sum_bits).
    """

    depth = len_lst*2+500
    sys.setrecursionlimit(depth)
    
    start_a = ''.join(['1']*iters + ['0']*(n_pools-iters))
    start_u = ''.join(['1']*(iters+1) + ['0']*(n_pools-iters-1))

    arrangement = hamiltonian_path_AU(size=len_lst, point = start_a, t = 'a', unions = [start_u])
    
    addresses = []
    for item in arrangement:
        address = []
        for i in range(len(item)):
            if item[i] == '1':
                address.append(i)
        addresses.append(address)
    #print(sum_bits(arrangement))
    return sum_bits(arrangement), addresses

def address_rearrangement_A(n_pools, iters, len_lst):
    
    """
    For A-hamiltonian path search.
    Takes number of pools, iters, and length of the path.
    Returns balance of the path and list of addresses.
    Is dependent on function(hamiltonian_path_A) and function(sum_bits).
    """
    
    depth = len_lst*2+500
    sys.setrecursionlimit(depth)

    vertices = []
    for combo in combinations(range(n_pools), iters):
        v = ['0']*n_pools
        for i in combo:
            v[i] = '1'
        vertices.append(''.join(v))
        
    G = {v: [] for v in vertices}
    for v1 in vertices:
        for v2 in vertices:
            if hamming_distance(v1, v2) == 2:
                G[v1].append(v2)
            
    arrangement = hamiltonian_path_A(G, len_lst, vertices[0])
    
    addresses = []
    for item in arrangement:
        address = []
        for i in range(len(item)):
            if item[i] == '1':
                address.append(i)
        addresses.append(address)
    #print(sum_bits(arrangement))
    return sum_bits(arrangement), addresses


# # Peptide overlap


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

def how_many_peptides(lst, ep_length):
    """
    Takes list of peptides and expected epitope length.
    Returns 1) Counter object with number of epitopes shared across number of peptides;
    2) dictionary with all possible epitopes as keys and in how many peptides thet are present as values.
    """

    sequence_counts = dict()
    counts = []

    for peptide in lst:
        for i in range(0, len(peptide) - ep_length + 1):
            sequence = peptide[i:i+ep_length]
            if sequence in sequence_counts.keys():
                sequence_counts[sequence] += 1
            else:
                sequence_counts[sequence] = 1

    for key in sequence_counts.keys():
        counts.append(sequence_counts[key])
    counts = Counter(counts)

    return counts, sequence_counts


# # Pooling


### Bad addresses search
def bad_address_predictor(all_ns):
    
    """
    Takes list of addresses, searches for three consecutive addresses with the same union, removes the middle one.
    Returns list of addresses.
    """
    
    wb = all_ns.copy()
    
    for i in range(len(wb)-1, 1, -1):
        n1 = wb[i]
        n2 = wb[i-1]
        n3 = wb[i-2]
        if set(n1 + n2) == set(n2 + n3) or set(n1 + n2) == set(n1 + n3):
            wb.remove(n2)
    return wb

### Pooling
def pooling(lst, addresses, n_pools):
    
    """
    Takes list of peptides, list of addresses, number of pools.
    Returns pools - list of peptides for each pool, and peptide_address - for each peptide its address.
    """
    
    pools = {key: [] for key in range(n_pools)}
    peptide_address = dict()

    for i in range(len(lst)):
        peptide = lst[i]
        peptide_pools = addresses[i]
        peptide_address[peptide] = peptide_pools
        for item in peptide_pools:
            pools[item].append(peptide)
    return pools, peptide_address


### Pools activation
def pools_activation(pools, epitope):
    
    """
    Takes peptide pooling scheme (pools) and epitope.
    Returns which pools will be activated given this epitope.
    Is used in function(run_experiment).
    """
    
    activated_pools = []
    for key in pools.keys():
        for item in pools[key]:
            if epitope in item:
                activated_pools.append(key)
                    
    activated_pools = list(set(activated_pools))              
    return activated_pools


### Epitope - activated pools table
def epitope_pools_activation(peptide_address, lst, ep_length):
    
    """
    Takes dictionary of peptide_addresses, list of peptides, epitope length.
    Returns activated pools for each possible epitope from peptides.
    Is used in function(run_experiment).
    """
    
    epitopes = []
    act_profile = dict()
    for item in lst:
        for i in range(len(item)):
            if len(item[i:i+ep_length]) == ep_length and item[i:i+ep_length] not in epitopes:
                epitopes.append(item[i:i+ep_length])
    for ep in epitopes:
        act = []
        for peptide in peptide_address.keys():
            if ep in peptide:
                act = act + list(peptide_address[peptide])
        act = sorted(list(set(act)))
        str_act = str(act)
        if str_act not in act_profile.keys():
            act_profile[str_act] = [ep]
        else:
            act_profile[str_act].append(ep)
    return act_profile

### Peptide determination
def peptide_search(lst, act_profile, act_pools, iters, n_pools, regime):
    
    """
    Takes activated pools and returns peptides and epitopes which led to their activation.
    Has two regimes: with and without dropouts.
    Is used in function(run_experiment).
    """
    
    if regime == 'without dropouts':
        act = str(sorted(list(act_pools)))
        epitopes = act_profile.get(act)
        if epitopes is not None:
            peptides = []
            for peptide in lst:
                if all(epitope in peptide for epitope in epitopes):
                    peptides.append(peptide)
            return peptides, epitopes
    elif regime == 'with dropouts':
        act = str(sorted(list(act_pools)))
        epitopes = act_profile.get(act)
        if len(act) == iters +1 and epitopes is not None:
            peptides = []
            for peptide in lst:
                if all(epitope in peptide for epitope in epitopes):
                    peptides.append(peptide)
            return peptides, epitopes
        else:
            rest = list(set(range(n_pools)) - set(act_pools))
            r = iters + 1 - len(act_pools)
            if r < 0:
                r = 0
            options = list(combinations(rest, r))
            possible_peptides = []
            possible_epitopes = []
            
            for option in options:
                act_try = act_pools + list(option)
                act_try = str(sorted(list(act_try)))
                epitopes = act_profile.get(act_try)
                if epitopes is not None:
                    possible_epitopes = possible_epitopes + epitopes
                    peptides = []
                    for peptide in lst:
                        if all(epitope in peptide for epitope in epitopes):
                            peptides.append(peptide)
                    possible_peptides = possible_peptides + peptides
            return list(set(possible_peptides)), list(set(possible_epitopes))

### Resulting table
def run_experiment(lst, peptide_address, ep_length, pools, iters, n_pools, regime):
    
    """
    Imitates experiment. Has two regimes: with and without dropouts.
    Takes list of peptides and runs experiment for every possible epitope.
    Returns activated pools, predicted peptides based on these activated pools.
    With dropouts imitates dropouts and returns number of possible peptides given each possible dropout combination.
    Is dependent on function(pools_activation), function(peptide_search), function(epitope_pools_activation).
    """
    
    act_profile = epitope_pools_activation(peptide_address, lst, ep_length)
    
    check_results = pd.DataFrame(columns = ['Peptide', 'Address', 'Epitope', 'Act Pools',
                                        '# of pools', '# of epitopes', '# of peptides', 'Remained', '# of lost',
                                           'Right peptide', 'Right epitope'])
    for peptide in lst:
        for i in range(len(peptide)):
            ep = peptide[i:i+ep_length]
            if len(ep) == ep_length:
                act = pools_activation(pools, ep)
                if regime == 'without dropouts':
                    peps, eps = peptide_search(lst=lst, act_profile=act_profile,
                                           act_pools = act,
                                           iters = iters, n_pools = n_pools,
                                           regime = 'without dropouts')
                    right_pep = str(peptide in peps)
                    right_ep = str(ep in eps)
                    row = {'Peptide':peptide, 'Address':str(peptide_address[peptide]), 'Epitope':ep,
                           'Act Pools':str(sorted(list(act))), '# of pools':len(act),
                           '# of epitopes':len(eps), '# of peptides':len(peps), 'Remained':'-', '# of lost':0,
                           'Right peptide':right_pep, 'Right epitope':right_ep}
                    check_results = pd.concat([check_results, pd.DataFrame(row, index = [0])])
                elif regime == 'with dropouts':
                    l = len(act)
                    for i in range(1, l+1):
                        lost = len(act) - i
                        lost_combs = list(combinations(act, i))
                        for lost_comb in lost_combs:
                            peps, eps = peptide_search(lst=lst, act_profile=act_profile,
                                           act_pools = list(lost_comb),
                                           iters = iters, n_pools = n_pools,
                                           regime = 'with dropouts')
                            right_pep = str(peptide in peps)
                            right_ep = str(ep in eps)
                
                            row = {'Peptide':peptide, 'Address':str(peptide_address[peptide]), 'Epitope':ep,
                                   'Act Pools':str(sorted(list(act))), '# of pools':len(act),
                                   '# of epitopes':len(eps), '# of peptides':len(peps),
                                   'Remained':str(list(lost_comb)), '# of lost':lost,
                                   'Right peptide':right_pep, 'Right epitope':right_ep}
                            check_results = pd.concat([check_results, pd.DataFrame(row, index = [0])])
    return check_results

## Functions for .stl files

def stl_generator(rows, cols, length, width, thickness, hole_radius, x_offset, y_offset, well_spacing,
                  coordinates):

    """
    Returns mesh object with generated 3D plate with necessary holes in coordinates.
    Is used in function(pools_stl).
    """

    hole_height = thickness + 2
    
    # Plate
    plate_mesh = trimesh.creation.box(extents=[length, width, thickness])
    translation = [length / 2, width / 2, thickness / 2]
    plate_mesh.apply_translation(translation)
    
    # Sets of coordinates
    batch_size = 10
    coordinate_batches = [coordinates[i:i + batch_size] for i in range(0, len(coordinates), batch_size)]

    for batch in coordinate_batches:
        # Empty mesh
        batch_mesh = None
        for coord in batch:
            i, j = coord[0]-1, coord[1]-1
            hole_x = x_offset + j * well_spacing
            hole_y = y_offset + i * well_spacing
            cylinder_mesh = trimesh.creation.cylinder(radius=hole_radius, height=hole_height)
            cylinder_mesh.apply_translation([hole_x, hole_y, thickness / 2])
        
            # Mesh + cylinders from set
            if batch_mesh is None:
                batch_mesh = cylinder_mesh
            else:
                batch_mesh = batch_mesh.union(cylinder_mesh, engine="scad")

        # Plate - mesh without cylinders
        plate_mesh = plate_mesh.difference(batch_mesh)
        
    return plate_mesh

def pools_stl(peptides_table, pools, rows = 16, cols = 24, length = 122.10, width = 79.97,
              thickness = 1.5, hole_radius = 4.0 / 2, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5):
    
    """
    Takes peptide pooling scheme.
    Returns dictionary with mesh objects (3D plate with holes), where one plate is one value, and its key is a pool index.
    Is dependent on function(stl_generator).
    """

    meshes_list = dict()

    for pool_N in set(pools.index):
        coordinates = []
        for peptide in pools['Peptides'].iloc[pool_N].split(';'):
            row_value = int([(x, peptides_table.columns[y]) for x, y in zip(*np.where(peptides_table.values == peptide))][0][0]+1)
            column_value = int([(x, peptides_table.columns[y]) for x, y in zip(*np.where(peptides_table.values == peptide))][0][1])
            coordinates.append([row_value, column_value])
        coordinates = coordinates + [[16, 24]]
        
        name = 'pool' + str(pool_N)
        
        m = stl_generator(rows, cols, length, width, thickness, hole_radius, x_offset, y_offset, well_spacing,
                 coordinates)
        meshes_list[name] = m
    return meshes_list

def zip_meshes_export(meshes_list):

    """
    Takes a dictionary with mesh objects.
    Exports a .zip file with stl files generated from these mesh objects.
    """

    zip_filename = 'Pools_stl.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for key in meshes_list.keys():
            stl_filename = f'{key}.stl'
            meshes_list[key].export(stl_filename)
            zipf.write(stl_filename)
            
def zip_meshes(meshes_list):

    """
    Takes a dictionary with mesh objects.
    Returns a .zip file with stl files generated from these mesh objects.
    """

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for key in meshes_list.keys():
            stl_buffer = BytesIO()
            meshes_list[key].export(stl_buffer, file_type='stl')
            stl_buffer.seek(0)
            zipf.writestr(f'{key}.stl', stl_buffer.read())
    zip_buffer.seek(0)
    return zip_buffer
