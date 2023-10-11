Usage
=============================

.. code-block:: python

   import combinatorial_peptide_pooling as cpp

To use the package for basic tasks, the **Quickstart** section is
enough. To read more about used functions, check other sections.

.. _quickstart-section:

Quickstart
----------

1. (Optional) **Check your peptide list for overlap consistency.**

   .. note:: Incosistent overlap length can lead to hindered results interpretation.

   You can check all peptides for their overlap length with the next
   peptide (list of peptides should be ordered):

   .. function:: cpp.all_overlaps (lst) -> Counter object
      :noindex:

      :param lst: ordered list of peptides
      :type lst: list
      :return: Counter object with the dictionary, where the key is the overlap length and the value is the number of pairs with such overlap.
      :rtype: Counter object

      .. code-block:: python

         >>> cpp.all_overlaps(lst)
         Counter({12: 251, 16: 1})


   => 251 pairs of peptides with an overlap of length of 12 amino acids,
   and 1 pair with an overlap of length 16 amino acids.

   Also, you can check which peptides have such an overlap with the next
   peptide:

   .. function:: cpp.find_pair_with_overlap (lst, target_overlap) -> list
      :noindex:

      :param lst: ordered list of peptides
      :type lst: list
      :param target_overlap: overlap length
      :type target_overlap: int
      :return: list of lists with peptides with specified overlap length.
      :rtype: list

      .. code-block:: python

         >>> cpp.find_pair_with_overlap(lst, 16)
         [['FDEDDSEPVLKGVKLHY', 'DEDDSEPVLKGVKLHYT']]

   => Overlap of length 16 amino acids is in peptides *FDEDDSEPVLKGVKLHY* and *DEDDSEPVLKGVKLHYT*.

   Also, you can check what number of peptides share the same epitope.
   It might help to interpret the results later.

   .. function:: cpp.how_many_peptides (lst, ep_length) -> Counter object, dictionary
      :noindex:

      :param lst: ordered list of peptides
      :type lst: list
      :param ep_length: expected epitope length
      :type ep_length: int
      :return:
         1) the Counter object with the number of epitopes shared across the number of peptides;
         2) the dictionary with all possible epitopes of expected length as keys and the number of peptides where these epitopes are present as values.
      :rtype: Counter object, dictionary

      .. code-block:: python

         >>> t, r = cpp.how_many_peptides(lst, 8)
         >>> t
         Counter({1: 6, 2: 1256, 3: 4})
         >>> r
         {'MFVFLVLL': 1,'FVFLVLLP': 1,VFLVLLPL': 1,'FLVLLPLV': 1,'LVLLPLVS': 1,'VLLPLVSS': 2, ...,}

   => There are 6 epitopes present in a single peptide, 1256 epitopes present shared by two peptides, and 4 epitopes shared by 4 peptides. For each epitope, number of peptides sharing it is in the dictionary.

2. (Optional) **Then you need to determine peptide occurrence across
   pools, i.e. to how many pools one peptide would be added.**

   .. note:: Peptide occurrence affects number of peptides in one pool, and therefore too high peptide occurrence may lead to higher dilution of a single peptide.

   .. function:: cpp.find_possible_k_values (n, l) -> list
      :noindex:

      :param n: number of pools
      :type n: int
      :param l: number of peptides
      :type l: int
      :return: list with possible peptide occurrences given number of pools and number of peptides.
      :rtype: Counter object, dictionary

      .. code-block:: python

         >>> cpp.find_possible_k_values(12, 250)
         [4, 5, 6, 7, 8]

   => Given 12 pools and 250 peptides, you can use peptide occurrence equal to 4, 5, 6, 7, 8.

   Choose one occurrence value appropriate for your task and proceed.

3. **Now, you need to find the address arrangement given your number of
   pools, number of peptides, and peptide occurrence.**

   We suggest you use the :func:`cpp.address_rearrangement_AU` function. In the section `Address arrangement <#arrangement-section>`_ you can find other functions that can perform such a task (based on Gray codes and on a trivial Hamiltonian path search).

   .. note:: With large parameters, the algorithm needs some time to finish the arrangement. If the arrangement fails, try with other parameters.

   .. function:: cpp.address_rearrangement_AU (n_pools, iters, len_lst) -> list, list
      :noindex:

      :param n_pools: number of pools
      :type n_pools: int
      :param iters: peptide occurrence
      :type iters: int
      :param len_lst: number of peptides
      :type len_lst: int
      :return:
         1) list with number of peptides in each pool;
         2) list with address arrangement
      :rtype: list, list

      .. code-block:: python

         >>> cpp.address_rearrangement_AU(n_pools=12, iters=4, len_lst=250)
         >>> b
         [81, 85, 85, 85, 81, 82, 87, 81, 85, 81, 84, 83]
         >>> lines
         [[0, 1, 2, 3],[0, 1, 3, 6],[0, 1, 6, 8],[1, 6, 8, 9],[6, 8, 9, 11], ... ]

   => You will get the expected number of peptides in each pool and address arrangement, which will be used in following steps.

4. **Now, you can distribute peptides across pools using the produced
   address arrangement. One peptide will be added to one produced
   address.**

   .. note:: Keep in mind that peptides should be ordered as they overlap.

   .. function:: cpp.pooling (lst, addresses, n_pools) -> dictionary, dictionary
      :noindex:

      :param lst: ordered list with peptides
      :type lst: list
      :param addresses: produced address arrangement
      :type addresses: list
      :param n_pools: number of pools
      :type n_pools: int
      :return:
         1) pools -- dictionary with keys as pools indices and values as peptides that should be added to this pools;
         2) peptide address -- dictionary with peptides as keys and corresponding addresses as values.
      :rtype: dictionary, dictionary

      .. code-block:: python

         >>> pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=12)
         >>> pools
         {0: ['MFVFLVLLPLVSSQCVN','VLLPLVSSQCVNLTTRT',VSSQCVNLTTRTQLPPA', ...], 1: ['MFVFLVLLPLVSSQCVN','VLLPLVSSQCVNLTTRT','TQDLFLPFFSNVTWFHA', ...], ... }
         >>> peptide_address
         {'MFVFLVLLPLVSSQCVN': [0, 1, 2, 3], 'VLLPLVSSQCVNLTTRT': [0, 1, 2, 10], ... }

   => You will get the pooling scheme and peptide addresses.

5. **Now, you can run the simulation using produced pools and peptide_address.**

   The simulation produces a DataFrame with every possible epitope of the provided length and all pools where this epitope is present. This table is needed to interpret the results.

   The function has two regimes: with and without drop-outs. Without
   drop-outs, it returns a table as there were no mistakes, and all
   pools that should be activated were activated. With drop-outs, it
   returns a table with all possible mistakes (i.e.all possible
   non-activated pools). This option will need time to be generated,
   usually several minutes, although it depends on the number of
   peptides and on occurrence.

   .. function:: cpp.run_experiment(lst, peptide_address, ep_length, pools, iters, n_pools, regime) -> pandas DataFrame
      :noindex:

      :param lst: ordered list with peptides
      :type lst: list
      :param peptide_address: peptides addresses produced by pooling
      :type peptide_address: dictionary
      :param ep_length: expected epitope length
      :type ep_length: int
      :param pools: pools produced by pooling
      :type pools: dictionary
      :param iters: peptide occurrence
      :type iters: int
      :param n_pools: number of pools
      :type n_pools: int
      :param regime: regime of simulation, with or without drop-outs
      :type regime: “with dropouts” or “without dropouts”
      :return:
         1) pools -- dictionary with keys as pools indices and values as peptides that should be added to this pools;
         2) peptide address -- dictionary with peptides as keys and corresponding addresses as values.
      :rtype: dictionary, dictionary

      .. code-block:: python

         >>> df = cpp.run_experiment(lst=lst, peptide_address=peptide_address, ep_length=8, pools=pools, iters=iters, n_pools=n_pools, regime='without dropouts')


   .. code-block:: python

      >>> df

   .. table::
      :widths: 10 10 10 10 10 10 10 10 10 10 10

      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | Peptide           | Address       | Epitope  | Act Pools        | # of pools | # of epitopes | # of peptides | Remained | # of lost | Right peptide | Right epitope |
      +===================+===============+==========+==================+============+===============+===============+==========+===========+===============+===============+
      | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]  | MFVFLVLL | [0, 1, 2, 3]     | 4          | 5             | 1             | --       | 0         | True          | True          |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]  | MFVFLVLL | [0, 1, 2, 3]     | 4          | 5             | 1             | --       | 0         | True          | True          |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | …                 |               |          |                  |            |               |               |          |           |               |               |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]  | VLLPLVSS | [0, 1, 2, 3, 10] | 5          | 5             | 2             | --       | 0         | True          | True          |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | …                 |               |          |                  |            |               |               |          |           |               |               |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | VLLPLVSSQCVNLTTRT | [0, 1, 2, 10] | VLLPLVSS | [0, 1, 2, 3, 10] | 5          | 5             | 2             | --       | 0         | True          | True          |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+
      | …                 |               |          |                  |            |               |               |          |           |               |               |
      +-------------------+---------------+----------+------------------+------------+---------------+---------------+----------+-----------+---------------+---------------+

   **Peptide** — peptide sequence

   **Address** — pool indices where this peptide should be added

   **Epitope** — checked epitope from this peptide

   **Act pools** — list with pool indices where this epitope is present

   **# of pools** — number of pools where this epitope is present

   **# of epitopes** — number of epitopes that are present in the same pools (= number of possible peptides upon activation of such pools)

   **# of peptides** — number of peptides in which there are epitopes that are present in the same pools (= number of possible peptides upon activation of such pools)

   **Remained** — only upon regime=”with dropouts”, list of pools remained after mistake

   **# of lost** — only upon regime=”with dropouts”, number of dropped pools due to mistake

   **Right peptide** — True or False, whether the peptide is present in the list of possible peptides

   **Right epitope** — True or False, whether the peptide is present in the list of possible peptides

   To interpret the results of the experiment, you need to find all rows
   where the “Act Pools” column contains your combination of activated
   pools. Then, you will know all possible peptides and epitopes that
   could lead to the activation of such a combination of pools.

   If you can not find your combination of activated pools in the table,
   here is the sequence of actions.

   After the experiment, you will know the number of activated pools.
   This number depends on the length of overlap and the length of the
   expected epitope. You can check the distribution of epitope presence
   in your peptides using :func:`cpp.how_many_peptides`
   function. The number of activated pools would be equal to peptide
   occurrence plus one per additional peptide sharing this epitope.

   This way, if the epitope is present only in 1 peptide (usually, it is
   the case for epitopes at the ends of the protein), then the number of
   activated pools is equal to peptide occurrence. If the epitope is
   present in two peptides, then the number of activated pools is equal
   to peptide occurrence +1.

   If overlap length is consistent across all peptides, then the number
   of activated pools would be the same for almost all epitopes (except
   for the epitopes at the ends of the protein). Although even if the
   overlap is inconsistent, you can use the analysis, but it will hinder
   the interpretation of the results in some cases.

   If a shift length between two peptides is equal to or less than the
   expected epitope length divided by two, then the number of activated
   pools should be equal to the peptide occurrence value + 1.

   If the number of activated pools is less than according to the rule
   described above, then three options are possible:

   -  The target peptide is the peptide at the end of your peptide list,
      and the target epitope is located not in an overlap of this
      peptide with the next one. This could be checked easily: if your
      activated pools are not the same as the activated pools for any
      epitope from the first or last peptide, then you should check our
      second option.
   -  For the target peptide, overlap with its neighbor is less than
      usual, and therefore target epitope is not shared by the usual
      number of peptides. You can check that using :func:`cpp.all_overlaps` or :func:`cpp.how_many_peptides`. Nevertheless, given the absence of drop-outs, you still should be able to find the target peptide in the table with simulation results by searching for all rows where the “Act Pools” column contains your combination of activated pools.
   -  Some pools were not activated, although they should be; then, we
      recommend using the “with drop-outs” regime of the simulation. It
      imitates drop-outs of all possible pools, so you should be able to
      find your case in the resulting table.

   If the number of activated pools is higher than according to the rule
   described above, then two options are possible:

   -  For the target peptide, overlap with its neighbor is bigger than
      usual, and therefore target epitope is shared between more
      peptides. You can check that using :func:`cpp.all_overlaps` or :func:`cpp.how_many_peptides`. Nevertheless, given the absence of drop-outs, you still should be able to find the target peptide in the table with simulation results by searching for all rows where the “Act Pools” column contains your combination of activated pools.
   -  Some pools were activated, although they should not be. This issue
      is not addressed in the package.

   .. code-block:: python

      >>> df = cpp.run_experiment(lst=lst, peptide_address=peptide_address, ep_length=8, pools=pools, iters=iters, n_pools=n_pools, regime='with dropouts')
      >>> df

   .. table::
      :widths: 10 10 10 10 10 10 10 10 10 10 10

      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | Peptide           | Address        | Epitope  | Act Pools         | # of pools | # of epitopes | # of peptides | Remained          | # of lost | Right peptide | Right epitope |
      +===================+================+==========+===================+============+===============+===============+===================+===========+===============+===============+
      | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]   | MFVFLVLL | [0, 1, 2, 3]      | 4          | 40            | 12            | [0, 1, 2]         | 1         | True          | False         |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]   | MFVFLVLL | [0, 1, 2, 3]      | 4          | 76            | 25            | [0, 1, 3]         | 1         | True          | False         |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | …                 |                |          |                   |            |               |               |                   |           |               |               |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | RTQLPPAYTNSFTRGVY | [8, 9, 10, 11] | RTQLPPAY | [0, 8, 9, 10, 11] | 5          | 5             | 2             | [0, 8, 9, 10, 11] | 0         | True          | True          |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | …                 |                |          |                   |            |               |               |                   |           |               |               |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | RTQLPPAYTNSFTRGVY | [8, 9, 10, 11] | TQLPPAYT | [0, 8, 9, 10, 11] | 5          | 190           | 53            | [8, 9]            | 3         | True          | True          |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+
      | ...               |                |          |                   |            |               |               |                   |           |               |               |
      +-------------------+----------------+----------+-------------------+------------+---------------+---------------+-------------------+-----------+---------------+---------------+

   **Peptide** — peptide sequence

   **Address** — pool indices where this peptide should be added

   **Epitope** — checked epitope from this peptide

   **Act pools** — list with pool indices where this epitope is present

   **# of pools** — number of pools where this epitope is present

   **# of epitopes** — number of epitopes that are present in the same pools
   (= number of possible peptides upon activation of such pools)

   **# of peptides** — number of peptides in which there are epitopes that
   are present in the same pools (= number of possible peptides upon
   activation of such pools)

   **Remained** — only upon regime=”with dropouts”, list of pools remained
   after mistake

   **# of lost** — only upon regime=”with dropouts”, number of dropped pools
   due to mistake

   **Right peptide** — True or False, whether the peptide is present in the list
   of possible peptides

   **Right epitope** — True or False, whether the peptide is present in the list
   of possible peptides

   **Right peptide** and **Right epitope** columns are needed to check the
   algorithm of dropped pool recovery. Either “Right peptide” or “Right
   epitope” should contain the value “True”; otherwise, recovery was
   unsuccessful.

   Also, the regime “with drop-outs” can not differentiate between
   dropped pools due to a mistake and absent pools due to experiment
   design. This way, for epitopes located at the end of proteins, the
   algorithm would think that pools were dropped and would try to
   recover them. Because of that, if you suspect the epitope located at
   the end of the peptide to be the target epitope, we recommend first
   using the “without drop-outs” regime. You can look at the sequence of
   actions described above. The same applies to peptides with longer
   overlap. So, we strongly recommend using peptides with consistent
   overlap length.

6. (Optional) **To avoid mixing pools manually, you can print special
   punch cards using files with their 3D models produced by this step.**

   One punch card is needed for each pool. Each punch card is a thin
   card with holes located at the spots where the needed peptides are
   located in the plate. Therefore, each punch card has the number of
   holes equal to the number of peptides in a pool. Then, this card
   should be placed on an empty tip box, and a tip should be inserted
   into each hole. This way, if you are using a multichannel pipette,
   all tips are already arranged to take only the required peptides.

   [The process you can look up here.]

   To generate the files with 3D models, you need two functions.

   .. note:: The rendering of 3D models is a long process, so it could take time.

   .. function:: cpp.pools_stl(peptides_table, pools, rows = 16, cols = 24, length = 122.10, width = 79.97, thickness = 1.5, hole_radius = 4.0 / 2, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5) -> dictionary
      :noindex:

      :param peptides_table: table representing the arrangement of peptides in a plate, is not produced by any function in the package
      :type peptides_table: pandas DataFrame
      :param pools: table with a pooling scheme, where one row represents each pool, pool index is the index column, and a string with all peptides added to this pool separated by “;” is “Peptides” column.
      :type pools: pandas DataFrame
      :param rows: int
      :type rows: int
      :param cols: number of columns in your plate with peptides
      :type cols: int
      :param length: length of the plate in mm
      :type length: float
      :param width: width of the plate in mm
      :type width: float
      :param thickness: desired thickness of the punch card, in mm
      :type thickness: float
      :param hole_radius: the radius of the holes, in mm, should be adjusted to fit your tip
      :type hole_radius: float
      :param x_offset: the margin along the X axis for the A1 hole, in mm
      :type x_offset: float
      :param y_offset: the margin along the Y axis for the A1 hole, in mm
      :type y_offset: float
      :param well_spacing: the distance between wells, in mm
      :type well_spacing: float
      :return: dictionary with Mesh objects, where key is pool index, and value is a Mesh object of a corresponding punch card.
      :rtype: dictionary

      .. code-block:: python

         >>> meshes_list = cpp.pools_stl(peptides_table, pools, rows = 16, cols = 24, length = 122.10, width = 79.97, thickness = 1.5, hole_radius = 2.0, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5)

   Now, you need to pass generated dictionary to the function exporting it as a .zip file.

   .. function:: cpp.zip_meshes_export(meshes_list) -> None
      :noindex:

      :param meshes_list: dictionary with Mesh objects, generated in previous step
      :type meshes_list: dictionary
      :return: export Mesh objects as STL files in .zip archive.
      :rtype: None

      .. code-block:: python

         >>> cpp.zip_meshes_export(meshes_list)

   => You will get a .zip archive with generated STL files. Then, you can send these STL files directly to a 3D printer. We recommend writing the index of the pool on the punch card. Also, you can check the generated STL files using OpenSCAD.

.. _occurrence-section:

Peptide occurrence search
------------------------------

factorial(num)

combination(n, k):

find_possible_k_values(n, l):

.. _arrangement-section:

Address arrangement
--------------------

find_q_r(n):

bgc(n, s = None):

n_bgc(n):

computing_ab_i_odd(s_2, l, v):

m_length_BGC(m, n):

gc_to_address(s_2, iters, n):

union_address(address, union):

address_union(address, union):

hamiltonian_path_AU(size, point, t, unions, path=None):

variance_score(bit_sums, s):

return_address_message(code, mode):

binary_union(bin_list):

hamming_distance(s1, s2):

sum_bits(arr):

hamiltonian_path_A(G, size, pt, path=None):

.. function:: cpp.address_rearrangement_AU (n_pools, iters, len_lst) -> list, list

      :param n_pools: number of pools
      :type n_pools: int
      :param iters: peptide occurrence
      :type iters: int
      :param len_lst: number of peptides
      :type len_lst: int
      :return:
         1) list with number of peptides in each pool;
         2) list with address arrangement
      :rtype: list, list

      .. code-block:: python

         >>> cpp.address_rearrangement_AU(n_pools=12, iters=4, len_lst=250)
         >>> b
         [81, 85, 85, 85, 81, 82, 87, 81, 85, 81, 84, 83]
         >>> lines
         [[0, 1, 2, 3],[0, 1, 3, 6],[0, 1, 6, 8],[1, 6, 8, 9],[6, 8, 9, 11], ... ]

address_rearrangement_A(n_pools, iters, len_lst):

.. _overlap-section:

Peptide overlap
--------------------

string_overlap(str1, str2):

.. function:: cpp.all_overlaps (lst) -> Counter object

      :param lst: ordered list of peptides
      :type lst: list
      :return: Counter object with the dictionary, where the key is the overlap length and the value is the number of pairs with such overlap.
      :rtype: Counter object

      .. code-block:: python

         >>> cpp.all_overlaps(lst)
         Counter({12: 251, 16: 1})

.. function:: cpp.find_pair_with_overlap (lst, target_overlap) -> list

      :param lst: ordered list of peptides
      :type lst: list
      :param target_overlap: overlap length
      :type target_overlap: int
      :return: list of lists with peptides with specified overlap length.
      :rtype: list

      .. code-block:: python

         >>> cpp.find_pair_with_overlap(lst, 16)
         [['FDEDDSEPVLKGVKLHY', 'DEDDSEPVLKGVKLHYT']]

.. function:: cpp.how_many_peptides (lst, ep_length) -> Counter object, dictionary

      :param lst: ordered list of peptides
      :type lst: list
      :param ep_length: expected epitope length
      :type ep_length: int
      :return:
         1) the Counter object with the number of epitopes shared across the number of peptides;
         2) the dictionary with all possible epitopes of expected length as keys and the number of peptides where these epitopes are present as values.
      :rtype: Counter object, dictionary

      .. code-block:: python

         >>> t, r = cpp.how_many_peptides(lst, 8)
         >>> t
         Counter({1: 6, 2: 1256, 3: 4})
         >>> r
         {'MFVFLVLL': 1,'FVFLVLLP': 1,VFLVLLPL': 1,'FLVLLPLV': 1,'LVLLPLVS': 1,'VLLPLVSS': 2, ...,}

.. _pooling-section:

Pooling and simulation
------------------------------

bad_address_predictor(all_ns):

.. function:: cpp.pooling (lst, addresses, n_pools) -> dictionary, dictionary

      :param lst: ordered list with peptides
      :type lst: list
      :param addresses: produced address arrangement
      :type addresses: list
      :param n_pools: number of pools
      :type n_pools: int
      :return:
         1) pools -- dictionary with keys as pools indices and values as peptides that should be added to this pools;
         2) peptide address -- dictionary with peptides as keys and corresponding addresses as values.
      :rtype: dictionary, dictionary

      .. code-block:: python

         >>> pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=12)
         >>> pools
         {0: ['MFVFLVLLPLVSSQCVN','VLLPLVSSQCVNLTTRT',VSSQCVNLTTRTQLPPA', ...], 1: ['MFVFLVLLPLVSSQCVN','VLLPLVSSQCVNLTTRT','TQDLFLPFFSNVTWFHA', ...], ... }
         >>> peptide_address
         {'MFVFLVLLPLVSSQCVN': [0, 1, 2, 3], 'VLLPLVSSQCVNLTTRT': [0, 1, 2, 10], ... }

pools_activation(pools, epitope):

epitope_pools_activation(peptide_address, lst, ep_length):

peptide_search(lst, act_profile, act_pools, iters, n_pools, regime):

.. function:: cpp.run_experiment(lst, peptide_address, ep_length, pools, iters, n_pools, regime) -> pandas DataFrame

      :param lst: ordered list with peptides
      :type lst: list
      :param peptide_address: peptides addresses produced by pooling
      :type peptide_address: dictionary
      :param ep_length: expected epitope length
      :type ep_length: int
      :param pools: pools produced by pooling
      :type pools: dictionary
      :param iters: peptide occurrence
      :type iters: int
      :param n_pools: number of pools
      :type n_pools: int
      :param regime: regime of simulation, with or without drop-outs
      :type regime: “with dropouts” or “without dropouts”
      :return:
         1) pools -- dictionary with keys as pools indices and values as peptides that should be added to this pools;
         2) peptide address -- dictionary with peptides as keys and corresponding addresses as values.
      :rtype: dictionary, dictionary

      .. code-block:: python

         >>> df = cpp.run_experiment(lst=lst, peptide_address=peptide_address, ep_length=8, pools=pools, iters=iters, n_pools=n_pools, regime='without dropouts')

.. _3D-section:

3D models
----------

stl_generator(rows, cols, length, width, thickness, hole_radius,
x_offset, y_offset, well_spacing, coordinates):

.. function:: cpp.pools_stl(peptides_table, pools, rows = 16, cols = 24, length = 122.10, width = 79.97, thickness = 1.5, hole_radius = 4.0 / 2, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5) -> dictionary

      :param peptides_table: table representing the arrangement of peptides in a plate, is not produced by any function in the package
      :type peptides_table: pandas DataFrame
      :param pools: table with a pooling scheme, where one row represents each pool, pool index is the index column, and a string with all peptides added to this pool separated by “;” is “Peptides” column.
      :type pools: pandas DataFrame
      :param rows: int
      :type rows: int
      :param cols: number of columns in your plate with peptides
      :type cols: int
      :param length: length of the plate in mm
      :type length: float
      :param width: width of the plate in mm
      :type width: float
      :param thickness: desired thickness of the punch card, in mm
      :type thickness: float
      :param hole_radius: the radius of the holes, in mm, should be adjusted to fit your tip
      :type hole_radius: float
      :param x_offset: the margin along the X axis for the A1 hole, in mm
      :type x_offset: float
      :param y_offset: the margin along the Y axis for the A1 hole, in mm
      :type y_offset: float
      :param well_spacing: the distance between wells, in mm
      :type well_spacing: float
      :return: dictionary with Mesh objects, where key is pool index, and value is a Mesh object of a corresponding punch card.
      :rtype: dictionary

      .. code-block:: python

         >>> meshes_list = cpp.pools_stl(peptides_table, pools, rows = 16, cols = 24, length = 122.10, width = 79.97, thickness = 1.5, hole_radius = 2.0, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5)

.. function:: cpp.zip_meshes_export(meshes_list) -> None

      :param meshes_list: dictionary with Mesh objects, generated in previous step
      :type meshes_list: dictionary
      :return: export Mesh objects as STL files in .zip archive.
      :rtype: None

      .. code-block:: python

         >>> cpp.zip_meshes_export(meshes_list)

zip_meshes(meshes_list):