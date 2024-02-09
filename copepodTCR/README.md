
# Recursive Combining Algorithm for Combinatorial Peptide Pooling Design of T-cell Receptors

The **Recursive Combining Algorithm (RCA)** is a powerful algorithm that can be employed in constructing sequences for combinatorial peptide pooling design of T-cell receptors (TCR). It proceeds by recursively combining elementary sequences into the base sequence, typically in a bottom-up fashion.

A detailed note on RCA is referred to **note_on_RCA.pdf**. 


## Implementation of RCA in Python
### Functions
* `address_rearrangement_RC` The exposure function for original RCA. It takes the number of pools, iters, and length of the sequence, and returns the address sequence. It is dependent on function (`find_path`).

* `address_rearrangement_RC2` The exposure function for advanced recursive combining (RCA+BBA). It takes number of pools, iters, and length of the path, and returns the balance of the sequence and list of addresses. It is dependent on function (`recursive_combining`).

* `recursive_combining` The core function for RCA. This function recursively calls itself, combines the augmented elementary sequences and terminates when achieving the desired depth.

* `gen_elementary_sequence` The function that generate augmented elementary sequences via BBA.

* `isGrayUnionDisjoint` The function that verify whether the constructed sequence satisfy the first two constraints (Gray and union disjoint).

* `change_row` Perform permutation to the augmented elementary sequence.

* `find_vector_distance_1` The function to find the vector b in each iteration to make sure the combined sequence in the next iteration will satisfy the first two constraints.

* `set_distance` The function that returns the distance of two addresses (in represent of a set).

* `union_adjacent_sets` The function that returns the union of adjacent addresses.

* `item_per_pool` The function that takes matrix of addresses and number of pools, and returns the balance.

* `find_path` The function that takes number of pools, pool per item and the directory of the elementary sequences, and returns the path of the elementary short sequence without shortening. It is used in original RCA.

### Tip
* In function `address_rearrangement_RC`, please change the directory of the pre-constructed short sequences to the path where it is downloaded.