#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import pandas as pd
import numpy as np
from itertools import combinations
import random
import warnings
from collections import Counter
import trimesh
import zipfile
from io import BytesIO
import pymc as pm
import arviz as az

import seaborn as sn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import plotly.graph_objects as go


# # setting seed
def set_seed(seed: int) -> None:
    """
    Sets seeds for Python random and NumPy random number generators.
    :param seed: random seed value
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)


# # functions for iters search

def factorial(num: int) -> int:
    """
    Calculates factorial of a number.
    :param num: number
    :return: factorial of num
    """

    if num == 0:
        return 1
    else:
        return num * factorial(num - 1)


def combination(n: int, k: int) -> int:
    """
    Calculates number of possible combinations.
    :param n: number of items
    :param k: number of selected items
    :return: number of combinations
    """

    return factorial(n) // (factorial(k) * factorial(n - k))


def find_possible_k_values(n: int, l: int) -> list[int]:
    """
    Finds possible peptide occurrences given number of pools and peptides.
    :param n: number of pools
    :param l: number of peptides
    :return: list of possible peptide occurrence values
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


# # peptide overlap

def peptide_generation(
    protein: str | list[str], peptide_length: int, peptide_shift: int,
    protein_end: bool = False
) -> list[str]:
    """
    Generates overlapping peptides from one protein or a list of proteins.
    :param protein: protein sequence or list of protein sequences
    :param peptide_length: length of each generated peptide
    :param peptide_shift: shift between consecutive peptides
    :param protein_end: whether to include a final shifted peptide near the C-terminus
    :return: list with generated peptide sequences

    If protein_end is True and a protein is shorter than peptide_length, no peptides are
    generated for this protein and a warning is raised.
    """

    peptide_lst = []

    if isinstance(protein, str):
        if protein_end == True and len(protein) < peptide_length:
            warnings.warn('Protein is shorter than peptide_length; no peptides were generated.')
            return peptide_lst
        for i in range(0, len(protein), peptide_shift):
            ps = protein[i:i + peptide_length]
            if len(ps) == peptide_length:
                peptide_lst.append(ps)
            else:
                if protein_end == True:
                    diff = peptide_length - len(ps)
                    ps = protein[i - diff:i - diff + peptide_length]
                    if len(ps) == peptide_length and peptide_lst[-1] != ps:
                        peptide_lst.append(ps)

    elif isinstance(protein, list):
        for protein_i, pr in enumerate(protein):
            if protein_end == True and len(pr) < peptide_length:
                warnings.warn(
                    f'Protein at index {protein_i} is shorter than peptide_length; '
                    f'no peptides were generated for this protein.'
                )
                continue
            for i in range(0, len(pr), peptide_shift):
                ps = pr[i:i + peptide_length]
                if len(ps) == peptide_length:
                    peptide_lst.append(ps)
                else:
                    if protein_end == True:
                        diff = peptide_length - len(ps)
                        ps = pr[i - diff:i - diff + peptide_length]
                        if len(ps) == peptide_length and peptide_lst[-1] != ps:
                            peptide_lst.append(ps)
    return peptide_lst


def string_overlap(str1: str, str2: str) -> int:
    """
    Calculates overlap length between two peptides.
    :param str1: first peptide sequence
    :param str2: second peptide sequence
    :return: overlap length
    """

    overlap_len = 0
    for i in range(1, min(len(str1), len(str2)) + 1):
        if str1[-i:] == str2[:i]:
            overlap_len = i
    return overlap_len


def all_overlaps(strings: list[str]) -> Counter[int]:
    """
    Counts overlap lengths between consecutive peptides.
    :param strings: ordered list of peptides
    :return: Counter with overlap lengths as keys and numbers of peptide pairs as values

    Raises a warning if more than one overlap length is detected.
    """

    overlaps = []
    for i in range(len(strings) - 1):
        overlaps.append(string_overlap(strings[i], strings[i + 1]))

    overlap_counts = Counter(overlaps)
    if len(overlap_counts) > 1:
        warnings.warn(f'Overlap length is inconsistent: {overlap_counts}.')

    return overlap_counts


def find_pair_with_overlap(strings: list[str], target_overlap: int) -> list[list[str]]:
    """
    Finds consecutive peptide pairs with a target overlap length.
    :param strings: ordered list of peptides
    :param target_overlap: overlap length to search for
    :return: list of peptide pairs with target overlap
    """

    target = []
    for i in range(len(strings) - 1):
        if string_overlap(strings[i], strings[i + 1]) == target_overlap:
            target.append([strings[i], strings[i + 1]])
    return target


def how_many_peptides(lst: list[str], ep_length: int) -> tuple[Counter[int], dict[str, int]]:
    """
    Counts how many peptides contain each possible epitope.
    :param lst: ordered list of peptides
    :param ep_length: expected epitope length
    :return:
        1) Counter with numbers of epitopes shared across numbers of peptides
        2) dictionary with epitopes as keys and numbers of containing peptides as values
    """

    sequence_counts = dict()
    counts = []

    for peptide in lst:
        for i in range(0, len(peptide) - ep_length + 1):
            sequence = peptide[i:i + ep_length]
            if sequence in sequence_counts.keys():
                sequence_counts[sequence] += 1
            else:
                sequence_counts[sequence] = 1

    for key in sequence_counts.keys():
        counts.append(sequence_counts[key])
    counts = Counter(counts)

    return counts, sequence_counts


# # pooling


### bad addresses search
def bad_address_predictor(all_ns: list[list[int]]) -> list[list[int]]:
    """
    Removes middle addresses from three consecutive addresses with the same union.
    :param all_ns: address arrangement
    :return: address arrangement with predicted bad middle addresses removed
    """

    wb = all_ns.copy()

    for i in range(len(wb) - 1, 1, -1):
        n1 = wb[i]
        n2 = wb[i - 1]
        n3 = wb[i - 2]
        if set(n1 + n2) == set(n2 + n3) or set(n1 + n2) == set(n1 + n3):
            wb.remove(n2)
    return wb


### pooling
def pooling(
    lst: list[str], addresses: list[list[int]], n_pools: int
) -> tuple[dict[int, list[str]], dict[str, list[int]]]:
    """
    Distributes peptides across pools using an address arrangement.
    :param lst: ordered list of peptides
    :param addresses: address arrangement
    :param n_pools: number of pools
    :return:
        1) dictionary with pool indices as keys and peptide lists as values
        2) dictionary with peptides as keys and pool addresses as values
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


### pools activation
def pools_activation(pools: dict[int, list[str]], epitope: str) -> list[int]:
    """
    Finds pools activated by an epitope.
    :param pools: peptide pooling scheme produced by pooling
    :param epitope: epitope sequence
    :return: list of activated pool indices
    """

    activated_pools = []
    for key in pools.keys():
        for item in pools[key]:
            if epitope in item:
                activated_pools.append(key)

    activated_pools = list(set(activated_pools))
    return activated_pools


### epitope-activated pools table
def epitope_pools_activation(
    peptide_address: dict[str, list[int]], lst: list[str], ep_length: int
) -> dict[str, list[str]]:
    """
    Builds activation profiles for all possible epitopes.
    :param peptide_address: peptide addresses produced by pooling
    :param lst: ordered list of peptides
    :param ep_length: expected epitope length
    :return: dictionary with stringified activated pools as keys and epitope lists as values
    """

    epitopes = []
    act_profile = dict()
    for item in lst:
        for i in range(len(item)):
            if len(item[i:i + ep_length]) == ep_length and item[i:i + ep_length] not in epitopes:
                epitopes.append(item[i:i + ep_length])
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

### peptide determination

def peptide_search(
    lst: list[str], ep_length: int, act_profile: dict[str, list[str]],
    act_pools: list[int], iters: int, n_pools: int, regime: str
) -> tuple[list[str], list[str]] | None:
    """
    Finds peptides and epitopes that explain activated pools.
    :param lst: ordered list of peptides
    :param ep_length: expected epitope length
    :param act_profile: activation profiles produced by epitope_pools_activation
    :param act_pools: activated pool indices
    :param iters: peptide occurrence across pools
    :param n_pools: number of pools
    :param regime: search regime, either 'with dropouts' or 'without dropouts'
    :return:
        1) possible peptides
        2) possible epitopes
    """

    c, _ = how_many_peptides(lst, ep_length)
    normal = max(c, key=c.get)

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
        if len(act_pools) == iters + normal - 1 and epitopes is not None:
            peptides = []
            for peptide in lst:
                if all(epitope in peptide for epitope in epitopes):
                    peptides.append(peptide)
            return peptides, epitopes
        else:
            rest = list(set(range(n_pools)) - set(act_pools))
            r = iters + normal - 1 - len(act_pools)
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

### resulting table
def run_experiment(
    lst: list[str], peptide_address: dict[str, list[int]], ep_length: int,
    pools: dict[int, list[str]], iters: int, n_pools: int, regime: str
) -> pd.DataFrame:
    """
    Simulates activated pools and predicted peptides for every possible epitope.
    :param lst: ordered list of peptides
    :param peptide_address: peptide addresses produced by pooling
    :param ep_length: expected epitope length
    :param pools: pools produced by pooling
    :param iters: peptide occurrence across pools
    :param n_pools: number of pools
    :param regime: simulation regime, either 'with dropouts' or 'without dropouts'
    :return: dataframe with activated pools and possible peptides for each epitope, with reset index
    """

    ## building the epitope activation profile used for all result rows
    act_profile = epitope_pools_activation(peptide_address, lst, ep_length)

    check_results = pd.DataFrame(
        columns=[
            'Peptide', 'Address', 'Epitope', 'Act Pools', '# of pools', '# of epitopes',
            '# of peptides', 'Remained', '# of lost', 'Right peptide', 'Right epitope'
        ]
    )

    for peptide in lst:
        for i in range(len(peptide)):
            ep = peptide[i:i + ep_length]
            if len(ep) == ep_length:
                act = pools_activation(pools, ep)
                if regime == 'without dropouts':
                    peps, eps = peptide_search(lst=lst, ep_length=ep_length, act_profile=act_profile,
                                               act_pools=act, iters=iters, n_pools=n_pools,
                                               regime='without dropouts')
                    right_pep = str(peptide in peps)
                    right_ep = str(ep in eps)
                    row = {
                        'Peptide': peptide, 'Address': str(peptide_address[peptide]), 'Epitope': ep,
                        'Act Pools': str(sorted(list(act))), '# of pools': len(act),
                        '# of epitopes': len(eps), '# of peptides': len(peps), 'Remained': '-',
                        '# of lost': 0, 'Right peptide': right_pep, 'Right epitope': right_ep
                    }
                    check_results = pd.concat([check_results, pd.DataFrame(row, index=[0])])
                elif regime == 'with dropouts':
                    l = len(act)
                    for i in range(1, l + 1):
                        lost = len(act) - i
                        lost_combs = list(combinations(act, i))
                        for lost_comb in lost_combs:
                            peps, eps = peptide_search(lst=lst, ep_length=ep_length, act_profile=act_profile,
                                                       act_pools=list(lost_comb), iters=iters, n_pools=n_pools,
                                                       regime='with dropouts')
                            right_pep = str(peptide in peps)
                            right_ep = str(ep in eps)

                            row = {
                                'Peptide': peptide, 'Address': str(peptide_address[peptide]), 'Epitope': ep,
                                'Act Pools': str(sorted(list(act))), '# of pools': len(act),
                                '# of epitopes': len(eps), '# of peptides': len(peps),
                                'Remained': str(list(lost_comb)), '# of lost': lost,
                                'Right peptide': right_pep, 'Right epitope': right_ep
                            }
                            check_results = pd.concat([check_results, pd.DataFrame(row, index=[0])])
    return check_results.reset_index(drop=True)

## functions for .stl files

def pick_engine() -> str:
    """
    Selects an available trimesh boolean operation engine.
    :return: available boolean operation engine name
    """

    import trimesh.boolean

    ## default engine is manifold3d
    try:
        import manifold3d  # noqa: F401
        return 'manifold'
    except ImportError:
        pass

    ## checking other available engines
    available = set(trimesh.boolean._engines.keys())

    if 'blender' in available:
        return 'blender'

    raise RuntimeError(
        f'No boolean backend available. Install manifold3d or Blender. '
        f'Available engines: {available}'
    )


def stl_generator(
    rows: int, cols: int, length: float, width: float, thickness: float,
    hole_radius: float, x_offset: float, y_offset: float, well_spacing: float,
    coordinates: list[tuple[int, int]], engine: str, marks: int | bool = False
) -> trimesh.Trimesh:
    """
    Generates a 3D plate mesh with holes at selected coordinates.
    :param rows: number of plate rows
    :param cols: number of plate columns
    :param length: plate length in mm
    :param width: plate width in mm
    :param thickness: mask thickness in mm
    :param hole_radius: hole radius in mm
    :param x_offset: X-axis offset for the first well in mm
    :param y_offset: Y-axis offset for the first well in mm
    :param well_spacing: distance between wells in mm
    :param coordinates: list of one-indexed row and column coordinates
    :param engine: trimesh boolean operation engine
    :param marks: number of pool-index marks to add, or False
    :return: trimesh mesh object
    """

    hole_height = thickness + 2

    ## creating the base plate before subtracting pool holes
    plate_mesh = trimesh.creation.box(extents=[length, width, thickness])
    plate_mesh.apply_translation([length / 2, width / 2, thickness / 2])

    ## splitting coordinates into batches for boolean operations
    batch_size = 40
    coordinate_batches = [coordinates[i:i + batch_size]
                          for i in range(0, len(coordinates), batch_size)]

    for batch in coordinate_batches:
        ## building all cylinders for this batch first
        cylinders = []
        for r, c in batch:
            i, j = r - 1, c - 1
            hole_x = x_offset + j * well_spacing
            hole_y = y_offset + i * well_spacing
            cyl = trimesh.creation.cylinder(radius=hole_radius, height=hole_height)
            cyl.apply_translation([hole_x, hole_y, thickness / 2])
            cylinders.append(cyl)

        ## uniting cylinders in one call and subtracting them from the plate
        if len(cylinders) == 1:
            batch_mesh = cylinders[0]
        else:
            batch_mesh = trimesh.boolean.union(
                cylinders, engine=engine, check_volume=False
            )

        # subtract the batch from the plate
        plate_mesh = plate_mesh.difference(
            batch_mesh, engine=engine, check_volume=False
        )

    ## adding optional pool index marks
    if marks:
        mark_meshes = []
        mark_space = 0
        for i in range(marks):
            y = well_spacing * 0.5
            x = well_spacing * 0.5 + i + mark_space
            mark_space += 1
            mark = trimesh.creation.box(extents=[1, 1, hole_height / 3])
            mark.apply_translation([x, y, thickness / 3])
            mark_meshes.append(mark)

        if mark_meshes:
            if len(mark_meshes) == 1:
                marks_union = mark_meshes[0]
            else:
                marks_union = trimesh.boolean.union(
                    mark_meshes, engine=engine, check_volume=False
                )
            plate_mesh = plate_mesh.difference(
                marks_union, engine=engine, check_volume=False
            )

    return plate_mesh


def pools_stl(
    peptides_table: pd.DataFrame, pools: pd.DataFrame, engine: str,
    rows: int = 16, cols: int = 24, length: float = 122.10, width: float = 79.97,
    thickness: float = 1.5, hole_radius: float = 4.0 / 2, x_offset: float = 9.05,
    y_offset: float = 6.20, well_spacing: float = 4.5, hole16: bool = False
) -> dict[str, trimesh.Trimesh]:

    """
    Generates 3D mask meshes for peptide pools.
    :param peptides_table: table with peptide positions in a plate
    :param pools: table with pool indices and semicolon-separated peptides in Peptides column
    :param engine: trimesh boolean operation engine
    :param rows: number of plate rows
    :param cols: number of plate columns
    :param length: plate length in mm
    :param width: plate width in mm
    :param thickness: mask thickness in mm
    :param hole_radius: hole radius in mm
    :param x_offset: X-axis offset for the first well in mm
    :param y_offset: Y-axis offset for the first well in mm
    :param well_spacing: distance between wells in mm
    :param hole16: whether to add a hole at position 16, 24
    :return: dictionary with pool names as keys and mesh objects as values
    """

    meshes_list = dict()

    for pool_N in set(pools.index):
        coordinates = []
        for peptide in pools['Peptides'].iloc[pool_N].split(';'):
            peptide_position = [
                (x, peptides_table.columns[y])
                for x, y in zip(*np.where(peptides_table.values == peptide))
            ][0]
            row_value = int(peptide_position[0] + 1)
            column_value = int(peptide_position[1])
            coordinates.append([row_value, column_value])
        if hole16:
            coordinates = coordinates + [[16, 24]]

        name = 'pool' + str(pool_N + 1)

        m = stl_generator(
            rows, cols, length, width, thickness, hole_radius, x_offset, y_offset,
            well_spacing, coordinates, marks=pool_N + 1, engine=engine
        )
        meshes_list[name] = m
    return meshes_list


def zip_meshes_export(meshes_list: dict[str, trimesh.Trimesh]) -> None:
    """
    Exports pool mask meshes to STL files and a zip archive.
    :param meshes_list: dictionary with mesh objects
    :return: None
    """

    zip_filename = 'Pools_stl.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for key in meshes_list.keys():
            stl_filename = f'{key}.stl'
            meshes_list[key].export(stl_filename)
            zipf.write(stl_filename)


def zip_meshes(meshes_list: dict[str, trimesh.Trimesh]) -> BytesIO:
    """
    Creates an in-memory zip archive with STL files generated from pool mask meshes.
    :param meshes_list: dictionary with mesh objects
    :return: BytesIO object containing a zip archive with STL files
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


# # bayesian model

### activation model
def activation_model(
    obs: list[float] | np.ndarray, n_pools: int, inds: list[int] | np.ndarray,
    neg_control: list[float] | np.ndarray | None = None, neg_share: float | None = None,
    cores: int = 1
) -> tuple[pm.Model, Axes | np.ndarray, pd.DataFrame, np.ndarray, az.InferenceData, list[float]]:
    """
    Fits a Bayesian mixture model to pool activation measurements.
    :param obs: observed activation values
    :param n_pools: number of pools
    :param inds: pool indices for observed activation values
    :param neg_control: negative control values; if None, values from the pool with the lowest mean signal are used
    :param neg_share: expected share of negative pools, default is 0.5
    :param cores: number of CPU cores for PyMC sampling
    :return:
        1) PyMC model object
        2) ArviZ posterior predictive axes
        3) dataframe with posterior mean probability of each pool being negative
        4) normalized negative control values used for model training
        5) full posterior sampling trace as an InferenceData object
        6) posterior means of the offset and negative component
    """

    coords = dict(pool=range(n_pools), component=('positive', 'negative'))
    obs = np.asarray(obs, dtype=float)
    inds = np.asarray(inds, dtype=int)

    if neg_share is None:
        neg_share = 0.5

    ## selecting negative control values for normalization
    if neg_control is None:
        pool_signal = pd.DataFrame({'Pool': inds, 'Signal': obs})
        lowest_signal_pool = pool_signal.groupby('Pool')['Signal'].mean().idxmin()
        neg_control = pool_signal.loc[pool_signal['Pool'] == lowest_signal_pool, 'Signal'].to_numpy()
    else:
        neg_control = np.asarray(neg_control, dtype=float)

    if np.min(neg_control) > np.max(obs):
        obs = obs / np.max(neg_control)
        neg_control = neg_control / np.max(neg_control)
    else:
        neg_control = neg_control / np.max(obs)
        obs = obs / np.max(obs)

    ## fitting a mixture model with negative and positive pool components
    with pm.Model(coords=coords) as model:

        negative = pm.TruncatedNormal('negative', mu=0, sigma=1, lower=0.0, upper=1.0)

        negative_obs = pm.TruncatedNormal('negative_obs', mu=negative, sigma=0.1, lower=0.0,
                                          upper=1.0, observed=neg_control)

        ## offset is constrained so negative + offset <= 1
        offset_proportion = pm.Beta('offset_proportion', alpha=5, beta=2)
        offset = pm.Deterministic('offset', (1 - negative) * offset_proportion)

        positive = pm.Deterministic('positive', negative + offset)

        p = pm.Beta('p', alpha=neg_share * 100, beta=(1 - neg_share) * 100)
        component = pm.Bernoulli('assign', p, dims='pool')

        mu_pool = negative * component + positive * (1 - component)

        sigma_neg = pm.HalfNormal('sigma_neg', 0.5)
        sigma_pos = pm.HalfNormal('sigm_pos', 0.2)
        sigma_pool = sigma_pos * (1 - component) + sigma_neg * component

        pool_dist = pm.TruncatedNormal('pool_dist', mu=mu_pool, sigma=sigma_pool, lower=0.0,
                                       upper=1.0, dims='pool')

        ## likelihood uses data indices to map observed values to pools
        sigma_data = pm.Exponential('sigma_data', 1.0)
        pm.TruncatedNormal('lik', mu=pool_dist[inds], sigma=sigma_data, observed=obs, lower=0.0, upper=1.0)

        idata_alt = pm.sample(cores=cores)

    with model:
        posterior_predictive = pm.sample_posterior_predictive(idata_alt)

    ax = az.plot_ppc(posterior_predictive, num_pp_samples=100,
                     colors=['#015396', '#FFA500', '#000000'])

    posterior = az.extract(idata_alt)
    n_mean = float(posterior['negative'].mean(dim='sample'))
    p_mean = float(posterior['offset'].mean(dim='sample'))

    posterior_p_mean = posterior['p'].mean(dim='sample').item()
    print(f'Posterior mean of p: {posterior_p_mean:.3f}')

    probs = posterior['assign'].mean(dim='sample').to_dataframe()

    ## only probs are important
    return model, ax, probs, neg_control, idata_alt, [p_mean, n_mean]


def peptide_probabilities(sim: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates probability of each peptide from simulation results and pool probabilities.
    :param sim: simulation table generated by run_experiment, with Address and Act Pools as list-like strings
    :param probs: dataframe with posterior pool probabilities generated by activation_model
    :return: dataframe with peptide probabilities, activated pool counts, non-activated pool counts, and reset index
    """

    sim_add = sim[['Peptide', 'Address', 'Act Pools']]
    sim_add = sim_add.drop_duplicates().copy()

    ## parsing pool indices saved in simulation output
    try:
        for i in range(len(sim_add)):
            sim_add.iloc[i, 1] = [int(i) for i in sim_add['Address'].iloc[i][1:-1].split(',')]
            sim_add.iloc[i, 2] = [int(i) for i in sim_add['Act Pools'].iloc[i][1:-1].split(',')]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            'Address and Act Pools columns should contain list-like strings, '
            'for example "[0, 1, 2]", produced by run_experiment or read back unchanged from its output.'
        ) from error

    sim_add['Probability'] = 0.0
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
                mul.append(1 - p)
                if p <= 0.5:
                    act.append(y)
                else:
                    non_act.append(y)
        probability = np.prod(mul)
        sim_add.iloc[i, 3] = probability
        sim_add.iloc[i, 4] = len(act)
        sim_add.iloc[i, 5] = len(non_act)
    sim_add['Probability'] = sim_add['Probability'] / sum(sim_add['Probability'])
    return sim_add.reset_index(drop=True)


def results_analysis(
    peptide_probs: pd.DataFrame, probs: pd.DataFrame, sim: pd.DataFrame
) -> tuple[int, str, list[str] | str, list[str]]:
    """
    Interprets peptide probabilities with model-classified pool activations.
    :param peptide_probs: peptide probability table produced by peptide_probabilities
    :param probs: posterior pool probability table produced by activation_model
    :param sim: simulation table produced by run_experiment
    :return:
        1) number of activated pools
        2) interpretation message
        3) most likely cognate peptide or peptides
        4) possible cognate peptides
    """

    ep_length = len(sim['Epitope'].iloc[0])
    all_lst = list(peptide_probs['Peptide'].drop_duplicates())

    ## modal number of peptides sharing the same epitope
    ## for epitopes longer than the overlap, this is usually 1
    c, _ = how_many_peptides(all_lst, ep_length)
    normal = max(c, key=c.get)

    act_pools = []
    for i in range(len(probs)):
        if probs['assign'].iloc[i] < 0.5:
            act_pools.append(i)

    end_peptides = [peptide_probs['Peptide'].iloc[0], peptide_probs['Peptide'].iloc[-1]]
    peptide_probs = peptide_probs.sort_values(by='Probability', ascending=False)

    ## checking whether top Normal peptides share an epitope
    topNormal = list(dict.fromkeys(list(peptide_probs['Peptide'])[:normal]))
    epitope_check = [False] * (len(topNormal) - 1)
    for i in range(len(topNormal[0])):
        check = topNormal[0][i:i + ep_length]
        for y in range(len(topNormal[1:])):
            if len(check) == ep_length and check in topNormal[1:][y]:
                epitope_check[y] = True
    epitope_check = all(epitope_check)

    ## checking whether top Normal results have pool drop-outs
    drop_check = [True] * len(topNormal)
    for i in range(len(topNormal)):
        check = peptide_probs['Non-Activated'][peptide_probs['Peptide'] == topNormal[i]].values[0]
        if check == 0:
            drop_check[i] = False
    drop_check = all(drop_check)

    ## checking whether top Normal peptides are located at the end of the protein
    end_check = [False]
    for p in topNormal:
        if p in end_peptides:
            end_check[0] = True
            end_check.append(p)

    peptide_address = dict()
    for p in all_lst:
        address = peptide_probs['Address'][peptide_probs['Peptide'] == p].iloc[0]
        peptide_address[p] = address

    ## interpreting edge cases before searching possible peptide explanations
    if len(act_pools) == len(probs):
        notification = 'All pools were activated'
        return len(act_pools), notification, [], []

    elif len(act_pools) == 0:
        notification = 'Zero pools were activated'
        return len(act_pools), notification, [], []

    elif epitope_check and not drop_check:
        notification = 'No drop-outs were detected'
        return len(act_pools), notification, topNormal, topNormal

    elif not epitope_check and drop_check:
        act_profile = epitope_pools_activation(peptide_address, all_lst, ep_length)
        iters = len(peptide_probs['Address'].iloc[0])
        n_pools = len(probs)
        peptides, epitopes = peptide_search(all_lst, ep_length, act_profile, act_pools, iters, n_pools,
                                            'with dropouts')
        if end_check[0]:
            notification = 'Cognate peptide is located at one of the ends of the list'
            return len(act_pools), notification, end_check[-1], peptides
        else:
            notification = 'Cognate peptides are not found'
            return len(act_pools), notification, [], peptides

    ## searching possible peptide explanations when drop-outs are detected
    elif epitope_check and drop_check:
        act_profile = epitope_pools_activation(peptide_address, all_lst, ep_length)
        iters = len(peptide_probs['Address'].iloc[0])
        n_pools = len(probs)
        act_number = iters + normal - 1
        if act_number > len(act_pools):
            notification = 'Drop-out was detected'
            peptides, epitopes = peptide_search(all_lst, ep_length, act_profile, act_pools, iters, n_pools,
                                                'with dropouts')
            return len(act_pools), notification, topNormal, peptides
        else:
            notification = 'False positive was detected'
            return len(act_pools), notification, [], []

    elif not epitope_check and not drop_check:
        act_profile = epitope_pools_activation(peptide_address, all_lst, ep_length)
        iters = len(peptide_probs['Address'].iloc[0])
        n_pools = len(probs)
        peptides, epitopes = peptide_search(all_lst, ep_length, act_profile, act_pools, iters, n_pools,
                                            'with dropouts')
        if end_check[0]:
            notification = 'Cognate peptide is located at one of the ends of the list'
            return len(act_pools), notification, end_check[-1], peptides
        else:
            notification = 'No drop-outs were detected'
            return len(act_pools), notification, topNormal, peptides
    else:
        notification = 'Analysis error'
        return len(act_pools), notification, [], []


# # simulated data

### peptides generation
def random_amino_acid_sequence(length: int) -> str:
    """
    Generates a random amino acid sequence.
    :param length: length of the generated amino acid sequence
    :return: random amino acid sequence
    """

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(random.choice(amino_acids) for _ in range(length))


### simulation
def simulation(
    mu_off: float, sigma_off: float, mu_n: float, sigma_n: float, r: int,
    sigma_p_r: float, sigma_n_r: float, n_pools: int, p_shape: int,
    pl_shape: int, low_offset: float, cores: int = 1
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """
    Simulates activation signal measurements for positive, low-positive, and negative pools.
    :param mu_off: mean offset between negative and positive signal distributions
    :param sigma_off: standard deviation of the offset and positive pool distributions
    :param mu_n: mean of the negative signal distribution
    :param sigma_n: standard deviation of the negative signal distribution
    :param r: number of replicates for each pool
    :param sigma_p_r: replicate noise for positive and low-positive pools
    :param sigma_n_r: replicate noise for negative pools
    :param n_pools: total number of pools
    :param p_shape: number of positive pools
    :param pl_shape: number of low-positive pools
    :param low_offset: positive signal multiplier for low-positive pools
    :param cores: number of CPU cores for PyMC sampling
    :return:
        1) simulated values for positive pools
        2) simulated values for low-positive pools
        3) simulated values for negative pools
        4) simulated negative control values
        5) posterior means of the offset and negative signal baseline
    """

    n_shape = n_pools - p_shape - pl_shape
    with pm.Model() as simulation:
        ## offset distribution
        offset = pm.TruncatedNormal('offset', mu=mu_off, sigma=sigma_off, lower=0, upper=100)

        ## negative and positive signal distributions
        n = pm.TruncatedNormal('n', mu=mu_n, sigma=sigma_n, lower=0, upper=100)
        raw_p = n + offset
        p = pm.Deterministic('p', pm.math.clip(raw_p, 0, 100))
        p_low = pm.Deterministic('p_low', p * low_offset)

        ## pool-level signal values before replicate noise
        n_pools = pm.TruncatedNormal('n_pools', mu=n, sigma=sigma_n, lower=0, upper=100, shape=n_shape)
        inds_n = list(range(n_shape)) * r
        n_shape_r = n_shape * r

        p_pools = pm.TruncatedNormal('p_pools', mu=p, sigma=sigma_off, lower=0, upper=100, shape=p_shape)
        inds_p = list(range(p_shape)) * r
        p_shape_r = p_shape * r

        pl_pools = pm.TruncatedNormal('pl_pools', mu=p_low, sigma=sigma_off, lower=0, upper=100, shape=pl_shape)
        inds_pl = list(range(pl_shape)) * r
        pl_shape_r = pl_shape * r

        ## adding replicate-level measurement noise
        p_pools_r = pm.TruncatedNormal('p_pools_r', mu=p_pools[inds_p], sigma=sigma_p_r, lower=0,
                                       upper=100, shape=p_shape_r)
        pl_pools_r = pm.TruncatedNormal('pl_pools_r', mu=pl_pools[inds_pl], sigma=sigma_p_r, lower=0,
                                        upper=100, shape=pl_shape_r)
        n_pools_r = pm.TruncatedNormal('n_pools_r', mu=n_pools[inds_n], sigma=sigma_n_r, lower=0,
                                       upper=100, shape=n_shape_r)

        ## simulating negative control values
        n_control = pm.TruncatedNormal('n_control', mu=n, sigma=sigma_n, lower=0, upper=100, shape=r)

        trace = pm.sample(draws=1, cores=cores)

    p_results = trace.posterior.p_pools_r.mean(dim='chain').values.tolist()[0]
    pl_results = trace.posterior.pl_pools_r.mean(dim='chain').values.tolist()[0]
    n_results = trace.posterior.n_pools_r.mean(dim='chain').values.tolist()[0]
    n_control = trace.posterior.n_control.mean(dim='chain').values.tolist()[0]

    n_mean = float(trace.posterior.n.mean())
    p_mean = float(trace.posterior.offset.mean())

    return p_results, pl_results, n_results, n_control, [p_mean, n_mean]


# # plotting results

def poolplot(
    probs: pd.DataFrame, cells: list[float], inds: list[int], most: list[str],
    ax: Axes | None = None
) -> Axes:
    """
    Plots pool activation measurements with model-classified activated pools in green.
    :param probs: dataframe with posterior pool probabilities generated by activation_model
    :param cells: observed activation values
    :param inds: pool indices for observed activation values
    :param most: most likely peptides generated by results_analysis
    :param ax: matplotlib axes for plotting, optional
    :return: matplotlib axes with the plot
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ## marking pools with posterior non-activated probability below 0.5
    inds_act = set(probs[probs['assign'] < 0.5].index)
    labels = ['act' if ind in inds_act else 'non-act' for ind in inds]
    palette = {'act': '#00A000', 'non-act': '#C1C1CA'}
    edgecolors = ['#00A000' if ind in inds_act else '#C1C1CA' for ind in inds]

    ax = sn.scatterplot(y=np.log10(cells), x=inds, s=100, hue=labels, alpha=0.5,
                        legend=False, palette=palette, edgecolors=edgecolors, ax=ax)

    if len(most) > 0:
        title = ' or '.join([str(peptide) for peptide in most])
        title_fontsize = 8
        if len(title) > 60:
            title_fontsize = 6
        if len(title) > 120:
            title_fontsize = 5
        ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel('Log10 of activated T cell percentage', fontsize=8)
    ax.set_xlabel('Pools', fontsize=14)
    ax.set_xticks(inds)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.figure.tight_layout()
    return ax


def bubbleplot(df: pd.DataFrame, peptide_shift: int = 5, ax: Axes | None = None) -> Axes:
    """
    Plots peptide probabilities as a bubble plot.
    :param df: dataframe with peptide probabilities generated by peptide_probabilities
    :param peptide_shift: shift between generated peptides, used to scale peptide position on X axis
    :param ax: matplotlib axes for plotting, optional
    :return: matplotlib axes with the plot
    """

    df = df.copy()
    df['s'] = (df['Activated'] - df['Non-Activated'])
    df = df[df['s'] > 0].reset_index(drop=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    jitter_strength = 0.2
    x_j = np.arange(len(df)) * peptide_shift + np.random.normal(0, jitter_strength, len(df))
    ax.scatter(x_j, (df['Probability']), s=df['s'] * 100, alpha=0.5, color='#00A000')
    ax.set_ylim(-0.05, max(df['Probability']) + 0.1)
    ax.set_xlabel('Peptide position in the protein')
    ax.set_ylabel('Peptide probability')
    ax.figure.tight_layout()
    return ax


def hover_bubbleplot(df: pd.DataFrame, peptide_shift: int = 5) -> go.Figure:
    """
    Plots interactive peptide probabilities as a bubble plot.
    :param df: dataframe with peptide probabilities generated by peptide_probabilities
    :param peptide_shift: shift between generated peptides, used to scale peptide position on X axis
    :return: interactive plotly figure
    """

    df['s'] = df['Activated'].astype(float) - df['Non-Activated'].astype(float)
    df = df[df['s'] > 0].reset_index(drop=True)

    n = len(df)
    jitter_strength = 0.2
    x_j = np.arange(n) * peptide_shift + np.random.normal(0, jitter_strength, n)

    sizes = (df['s'] * 10.0).to_numpy()
    y = df['Probability'].astype(float).to_numpy()

    ## preparing custom peptide and address data for plotly hover text
    peptide = df['Peptide'].astype(str).to_numpy()
    address = df['Address'].apply(lambda v: str(v)).to_numpy()
    custom = np.column_stack([peptide, address])

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_j,
                y=y,
                mode='markers',
                marker=dict(size=sizes, opacity=0.5, color='#00A000'),
                customdata=custom,
                hovertemplate=(
                    'Peptide: %{customdata[0]}<br>'
                    'Address: %{customdata[1]}<br>'
                    'Probability: %{y:.4f}<br>'
                    '<extra></extra>'
                ),
            )
        ]
    )

    fig.update_layout(
        width=1000,
        height=350,
        xaxis_title='Peptide position in the protein',
        yaxis_title='Peptide probability',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            showgrid=False,
            zeroline=False
        )
    )
    fig.update_yaxes(range=[-0.05, float(df['Probability'].max()) + 0.1])

    return fig
