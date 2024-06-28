#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from itertools import combinations
import cvxpy as cp
import math
import random
from collections import Counter
import trimesh
import sys
from io import StringIO
import zipfile
from io import BytesIO
import pymc as pm
import arviz as az


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


# # Bayesian Model

### Activation model
def activation_model(obs, n_pools, inds, neg_control = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))

    obs = obs/np.max(obs)
    if neg_control is not None:
        neg_control = neg_control/np.max(obs)
    else:
        neg_control = 0.5/np.max(obs)

    with pm.Model(coords=coords) as alternative_model:
        # Define the offset
        offset = pm.Normal("offset", mu=0.6, sigma=0.1)

        # Negative component remains the same
        source_negative = pm.TruncatedNormal(
            "negative",
            mu=0,
            sigma=0.01,
            lower=0,
            upper=1,
            )
        # Adjusted positive distribution with offset
        source_positive = pm.Deterministic("positive", source_negative + offset)

        # Combine the source components
        source = pm.math.stack([source_positive, source_negative], axis=0)

        # Probability of assigning depends on negative control
        neg_sigma = np.sqrt(neg_control * (1 - neg_control))/2
        #prior = pm.TruncatedNormal('prior', mu = neg_control, sigma = neg_sigma, lower=0, upper=1)
        prior = pm.Beta('prior', mu = neg_control, sigma = neg_sigma)

        # Each pool is assigned a 0/1 (could adjust the prior probability here given that we
        # know negatives are more likely a priori)
        component = pm.Bernoulli("assign", 1-prior, dims="pool")

        # Each pool has a normally distributed response whose mu comes from either the
        # postive or negative source distribution
        pool_dist = pm.TruncatedNormal(
            "pool_dist",
            mu=source[component]+neg_sigma,
            sigma=pm.Exponential("sigma", 1),
            lower=0,
            upper=1,
            dims="pool",
            )

        # Likelihood, where the data indices pick out the relevant pool from pool
        pm.TruncatedNormal(
            "lik",
            mu=pool_dist[inds],
            sigma=pm.Exponential("sigma_data", 1),
            observed=obs,
            lower=0,
            upper=1,
            )

        idata_alt = pm.sample(cores = cores)
        
    with alternative_model:
        posterior_predictive = pm.sample_posterior_predictive(idata_alt)

    ax = az.plot_ppc(posterior_predictive, num_pp_samples=100, colors = ['#015396', '#FFA500', '#000000'])

    posterior = az.extract(idata_alt)
    n_mean = posterior["negative"].mean(dim="sample")
    p_mean = posterior["positive"].mean(dim="sample")

    #ax2 = pm.model_to_graphviz(alternative_model)
    #print(posterior["assign"].mean(dim="sample").to_dataframe())
    return ax, posterior["assign"].mean(dim="sample").to_dataframe(), [p_mean, n_mean]

def peptide_probabilities(sim, probs):

    """
    Takes a dataframe with probabilities (generated by function(activation_model)) and simulation without drop-outs (generated by function(run_experiment)).
    Returns a probability of each peptide in a DataFrame.
    """
    
    sim_add = sim[['Peptide', 'Address', 'Act Pools']]
    sim_add = sim_add.drop_duplicates()
    for i in range(len(sim_add)):
        sim_add['Act Pools'].iloc[i] = [int(i) for i in sim_add['Act Pools'].iloc[i][1:-1].split(',')]
        sim_add['Address'].iloc[i] = [int(i) for i in sim_add['Address'].iloc[i][1:-1].split(',')]
    sim_add['Probability'] = 0.1
    sim_add['Activated'] = 0
    sim_add['Non-Activated'] = 0
    
    for i in range(len(sim_add)):
        ad = sim_add.iloc[i, 2]
        mul = []
        act = []
        non_act = []
        for y in range(len(probs)):
            p = probs['assign'].iloc[y]
            if y not in ad:
                mul.append(p)
            else:
                mul.append(1-p)
                if p <= 0.5:
                    act.append(y)
                else:
                    non_act.append(y)
        probability = np.prod(mul)
        sim_add.iloc[i, 3] = probability
        sim_add.iloc[i, 4] = len(act)
        sim_add.iloc[i, 5] = len(non_act)
    sim_add['Probability'] = sim_add['Probability']/sum(sim_add['Probability'])
    #sim_add = sim_add.sort_values(by = 'Probability', ascending = False)
    return sim_add

def results_analysis(peptide_probs, probs, sim):

    """
    Takes a dataframe with probabilities (generated by function(activation_model)), simulation without drop-outs (generated by function(run_experiment)), and probabilities for each peptide generated by function(peptide_probabilities).
    Returns resulting peptides.
    """
    
    ep_length = len(sim['Epitope'].iloc[0])
    all_lst = list(peptide_probs['Peptide'].drop_duplicates())
    c, _ = how_many_peptides(all_lst, ep_length)
    normal = max(c, key=c.get)
    
    act_pools = []
    for i in range(len(probs)):
        if probs['assign'].iloc[i] < 0.5:
            act_pools.append(i)
    
    peptide_probs = peptide_probs.sort_values(by = 'Probability', ascending=False)

     ## Whether top Normal peptides share an epitope
    lst = list(set(list(peptide_probs['Peptide'])[:normal]))
    epitope_check = [False]*(len(lst)-1)
    for i in range(len(lst[0])):
        check = lst[0][i:i+ep_length]
        for y in range(len(lst[1:])):
            if len(check) == ep_length and check in lst[1:][y]:
                epitope_check[y] = True
    epitope_check = all(epitope_check)
    
    ## Whether top Normal results do not have drop outs
    drop_check = [False]*len(lst)
    for i in range(len(lst)):
        check = peptide_probs['Non-Activated'][peptide_probs['Peptide'] == lst[i]].values[0]
        if check == 0:
            drop_check[i] = True
    drop_check = all(drop_check)

    peptide_address = dict()
    for p in all_lst:
        address = peptide_probs['Address'][peptide_probs['Peptide'] == p].iloc[0]
        peptide_address[p] = address

    ## If all pools are marked as activated, then the results are compromised
    if len(act_pools) == len(probs):
        notification = 'Zero pools were activated'
        return notification, [], []

    ## If both are True, then the epitope is found:
    if all([drop_check, epitope_check]) == True:
        notification = 'No drop-outs were detected'
        return notification, lst, lst
        
    ## If epitope_check is held, but drop_check is not
    elif epitope_check == True and drop_check != True:
        ## Calculation of possible peptides
        act_profile = epitope_pools_activation(peptide_address, all_lst, ep_length)
        iters = len(peptide_probs['Address'].iloc[0])
        n_pools = len(probs)
        act_number = iters + normal -1
        if act_number > len(act_pools):
            notification = 'Drop-out was detected'
            peptides, epitopes = peptide_search(all_lst, act_profile, act_pools, iters, n_pools, 'with dropouts')
            return notification, lst, peptides
        else:
            notification = 'False positive was detected'
            return notification, [], []
            
    elif epitope_check != True and drop_check != True:
        ## More drop-outs happened, calculation of possible peptides
        act_profile = epitope_pools_activation(peptide_address, all_lst, ep_length)
        iters = len(peptide_probs['Address'].iloc[0])
        n_pools = len(probs)
        act_number = iters + normal -1
        peptides, epitopes = peptide_search(all_lst, act_profile, act_pools, iters, n_pools, 'with dropouts')
        if len(peptides) == 0:
            notification = 'Not found'
            return notification, [], peptides
        else:
            notification = 'Drop-out was detected'
            return notification, [], peptides

# # Simulated data

###Peptides generation
def random_amino_acid_sequence(length):
    '''
    Takes the length (integer).
    Returns random amino acid sequence of desired length.
    '''
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(random.choice(amino_acids) for _ in range(length))

### Simulation
def simulation(mu_off, sigma_off, mu_n, sigma_n, r, sigma_p_r, sigma_n_r, n_pools, p_shape,
               pl_shape, low_offset, cores=1):

    n_shape = n_pools-p_shape-pl_shape
    with pm.Model() as simulation:
        # offset
        offset = pm.Normal("offset", mu=mu_off, sigma=sigma_off)
    
        # Negative
        n = pm.TruncatedNormal('n', mu=mu_n, sigma=sigma_n, lower=0, upper=100)
        # Positive
        p = pm.Deterministic("p", n + offset)
        # Low positive
        p_low = pm.Deterministic("p_low", p*low_offset)

        # Negative pools
        n_pools = pm.TruncatedNormal('n_pools', mu=n, sigma=sigma_n, lower=0, upper=100, shape = n_shape)
        inds_n = list(range(n_shape))*r
        n_shape_r = n_shape*r

        # Positive pools
        p_pools = pm.TruncatedNormal('p_pools', mu=p, sigma=sigma_off, lower=0, upper=100, shape = p_shape)
        inds_p = list(range(p_shape))*r
        p_shape_r = p_shape*r

        # Low positive pools
        pl_pools = pm.TruncatedNormal('pl_pools', mu=p_low, sigma=sigma_off, lower=0, upper=100, shape = pl_shape)
        inds_pl = list(range(pl_shape))*r
        pl_shape_r = pl_shape*r

        # With replicas
        p_pools_r = pm.TruncatedNormal('p_pools_r', mu=p_pools[inds_p], sigma=sigma_p_r, lower=0, upper=100, shape=p_shape_r)
        pl_pools_r = pm.TruncatedNormal('pl_pools_r', mu=pl_pools[inds_pl], sigma=sigma_p_r, lower=0, upper=100, shape=pl_shape_r)
        n_pools_r = pm.TruncatedNormal('n_pools_r', mu=n_pools[inds_n], sigma=sigma_n_r, lower=0, upper=100, shape=n_shape_r)

        trace = pm.sample(draws=1, cores = cores)
        
    p_results = trace.posterior.p_pools_r.mean(dim="chain").values.tolist()[0]
    pl_results = trace.posterior.pl_pools_r.mean(dim="chain").values.tolist()[0]
    n_results = trace.posterior.n_pools_r.mean(dim="chain").values.tolist()[0]

    n_mean = float(trace.posterior.n.mean())
    p_mean = float(trace.posterior.p.mean())

    return p_results, pl_results, n_results, [p_mean, n_mean]