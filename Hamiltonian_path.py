#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from itertools import combinations
import cvxpy as cp
import random
from collections import Counter

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