# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:47:05 2024

@author: HEGUANCHEN
"""




import functions as func

n_pools=14
iters=3
len_lst=364

deviations_RC = []
deviations_A = []

times_RC = []
times_A = []

# for len_lst in range(500, 4001, 100):
    
print(len_lst)
# S_RC = func.address_rearrangement_RC2(n_pools=n_pools, iters=iters, len_lst=len_lst)
S_RC = func.address_rearrangement_RC(n_pools=n_pools, iters=iters, len_lst=len_lst)
balance_RC = func.item_per_pool(S_RC, n_pools)
deviation_RC= max(balance_RC) - min(balance_RC)
flag_RC = func.isGrayUnionDisjoint(S_RC)
deviations_RC.append(deviation_RC)