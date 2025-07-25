Usage
=============================

.. code-block:: python

   import copepodTCR as cpp


To use the package for basic tasks, the **Quickstart** section is
enough. To read more about used functions, check other sections.

.. _quickstart-section:

Quickstart
----------

.. code-block:: python

   import copepodTCR as cpp
   import codepub as cdp

   # number of pools
   n_pools = 12
   # peptide occurrence
   iters = 4
   # number of peptides
   len_lst = 253

   # address arrangemement
   b, lines = cdp.bba(m=n_pools, r=iters, n=len_lst)

   # if you have a lot of peptide, for faster address arrangement we recommend using:
   ## b, lines = cdp.rcbba(m=n_pools, r=iters, n=len_lst)

   # add your peptides to lst
   lst = list(pd.read_csv('peptides.csv', sep = "\t"))

   # pooling scheme generation
   pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=n_pools)

   # simulation
   check_results = cpp.run_experiment(lst=lst, peptide_address=peptide_address, ep_length=8, pools=pools, iters=iters, n_pools=n_pools, regime='without dropouts')

   # STL files generation
   # add peptide scheme to peptides_table_stl, with header and index as column and row numbers
   peptides_table_stl = pd.read_csv('peptides_scheme.tsv', sep = "\t", index_col = 0)
   pools_df = pd.DataFrame({'Peptides': [';'.join(val) for val in pools.values()]}, index=pools.keys())
   meshes_list = cpp.pools_stl(peptides_table = peptides_table_stl, pools = pools_df, rows = 16, cols = 24, length = 122.10, width = 79.97,
              thickness = 1.5, hole_radius = 2, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5)
   cpp.zip_meshes_export(meshes_list)

   # Results of the experiment as a table with two columns, Pool and Percentage. Activation signal is expressed in percentaged of activated T cells.
   exp_results = pd.read_csv('path/to/your/file')
   cells = list(exp_results['Percentage'])
   inds = list(exp_results['Pool'])

   # Model
   fig, probs = cpp.activation_model(cells, n_pools, inds)
   peptide_probs = cpp.peptide_probabilities(sim, probs)
   message, most, possible = cpp.results_analysis(peptide_probs, probs, check_results)
   print(message)
   print(most)
   print(possible)

.. _quickstartf-section:

More detailed quickstart
----------------------------------------

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

   For that, we developed a separate package **codepub**.

   We suggest you use the **codepub.bba** function. If you have a lot of peptides (1000+), we recommend using faster alternative: **codepub.rcbba**. Codepub documentation is available here: `CodePUB readthedocs <https://codepub.readthedocs.io/en/latest/Introduction.html>`_

   .. note:: With large parameters, the algorithm needs some time to finish the arrangement. If the arrangement fails, try with other parameters.

   .. function:: bba(m, r, n, start_a = None, W_des = None) -> list, list

      .. note:: Search for arrangement may take some time, especially with large parameters. This function is **slower** than :func:`cdp.rcbba`, but is more reliable.

      :param m: number of pools
      :type m: int
      :param r: address weight, i.e. to how many pools one item is added
      :type r: int
      :param n: number of items
      :type n: int
      :param start_a: desired first address of the arrangement, optional
      :type start_a: str
      :param W_des: desired balance for the resulting arrangement
      :type W_des: 
      :return:
         1) list with number of item in each pool, i.e. balance;
         2) list with address arrangement
      :rtype: list, list

      .. code-block:: python

         >>> balance, lines = cdp.bba(n_pools=12, iters=4, len_lst=250)
         >>> balance
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

      .. note:: Simulation may take several minutes, especially upon "with drop-outs" regime.

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
         pandas DataFrame with all possible epitopes of given length and the resulting activated pools
      :rtype: pandas DataFrame

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

7. **To interpret the results, you can use the Bayesian mixture model of activation signal.**
   
   Plate notation for the model (for 12 pools and 3 replicas).

   .. image:: model_scheme.png

   .. function:: cpp.activation_model(obs, n_pools, inds, neg_control=None, neg_share=None, cores=1) -> model, fig, pandas DataFrame, list, InferenceData, list
      :noindex:

      .. note:: Fitting might take several minutes.

      :param obs: list with observed values
      :type obs: list
      :param n_pools: number of pools
      :type n_pools: int
      :param inds: list with indices for observed values
      :type inds: list
      :param neg_control: optional list with negative control values; if not provided, it is estimated from obs
      :type neg_control: list or None
      :param neg_share: expected share of negative pools (between 0 and 1); default is 0.5
      :type neg_share: float or None
      :param cores: number of CPU cores to use for MCMC sampling
      :type cores: int
      :return:
         1) model -- PyMC model object used for fitting  
         2) fig -- posterior predictive KDE and observed data KDE (ArviZ)
         3) probs -- probability for each pool of being drawn from a distribution of activated or non-activated pools
         4) neg_control -- normalized control values used in model
         5) idata_alt -- full posterior sampling trace (InferenceData object)
         6) [p_mean, n_mean] -- posterior mean of the offset and baseline (negative) component
      :rtype: model, figure, pandas DataFrame, list, arviz.InferenceData, list

      .. code-block:: python

         >>> model, fig, probs, neg_control, trace, [p_mean, n_mean] = cpp.activation_model(obs, 12, inds)

      .. image:: model_fit.png

      .. code-block:: python

         >>> probs

      .. table::
         :widths: 10 10

         +------+---------+
         | Pool | assign  |
         +======+=========+
         | 0    | 0.99900 |
         +------+---------+
         | 1    | 1.00000 |
         +------+---------+
         | 2    | 0.00025 |
         +------+---------+
         | 3    | 0.36475 |
         +------+---------+
         | 4    | 0.00025 |
         +------+---------+
         | 5    | 0.00000 |
         +------+---------+
         | 6    | 1.00000 |
         +------+---------+
         | 7    | 1.00000 |
         +------+---------+
         | 8    | 0.99975 |
         +------+---------+
         | 9    | 0.99975 |
         +------+---------+
         | 10   | 0.00000 |
         +------+---------+
         | 11   | 0.99975 |
         +------+---------+


   The **Pool** column contains pool index, and column **assign** the probability of the pools to be drawn from the distribution of non-activated pool. The pool is considered to be activated if assign <= 0.5.

   Using this table, you can assess which pools were activated and which were not, and then check the result in check_results table with simulation. However, also you can use the following functions:

   .. function:: cpp.peptide_probabilities(sim, probs) -> pandas DataFrame
      :noindex:

      :param sim: check_results table with simulation with or without drop-outs
      :type sim: pandas DataFrame
      :param probs: DataFrame with probabilities produced by :func:`cpp.activation_model`
      :type probs: pandas DataFrame
      :return: peptide_probs -- probabilitity for each peptide to cause such a pattern of activation
      :rtype: pandas DataFrame

      .. code-block:: python

         >>> peptide_probs = cpp.peptide_probabilities(sim, probs)

      .. code-block:: python

         >>> peptide_probs

      .. table::
         :widths: 10 10 10 10 10 10

         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | Peptide           | Address       | Act Pools          | Probability  | Activated | Non-Activated |
         +===================+===============+====================+==============+===========+===============+
         | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]  | [0, 1, 2, 3]       | 1.172135e-07 | 2         | 2             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | MFVFLVLLPLVSSQCVN | [0, 1, 2, 3]  | [0, 1, 2, 3, 7]    | 8.262788e-10 | 2         | 2             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | VLLPLVSSQCVNLTTRT | [1, 2, 3, 7]  | [0, 1, 2, 3, 7]    | 8.262788e-10 | 2         | 2             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | VLLPLVSSQCVNLTTRT | [1, 2, 3, 7]  | [1, 2, 3, 7, 11]   | 2.119434e-05 | 3         | 3             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | VSSQCVNLTTRTQLPPA | [2, 3, 7, 11] | [1, 2, 3, 7, 11]   | 2.119434e-05 | 3         | 3             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | ...               | ...           | ...                | ...          | ...       | ...           |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | FDEDDSEPVLKGVKLHY | [0, 1, 3, 5]  | [0, 1, 2, 3, 4, 5] | 3.259596e-08 | 3         | 3             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | FDEDDSEPVLKGVKLHY | [0, 1, 3, 5]  | [0, 1, 2, 3, 5]    | 2.104844e-06 | 3         | 2             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | DEDDSEPVLKGVKLHYT | [0, 1, 2, 5]  | [0, 1, 2, 3, 4, 5] | 3.259596e-08 | 3         | 3             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | DEDDSEPVLKGVKLHYT | [0, 1, 2, 5]  | [0, 1, 2, 3, 5]    | 2.104844e-06 | 3         | 2             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+
         | DEDDSEPVLKGVKLHYT | [0, 1, 2, 5]  | [0, 1, 2, 5]       | 7.922877e-09 | 2         | 2             |
         +-------------------+---------------+--------------------+--------------+-----------+---------------+

   And then this table can be used to find cognate peptides:

   .. function:: cpp.results_analysis(peptide_probs, probs, sim) -> list, list, list
      :noindex:

      :param peptide_probs: DataFrame with probabilities for each peptide produced by :func:`cpp.peptide_probabilities`
      :type peptide_probs: pandas DataFrame
      :param probs: DataFrame with probabilities produced by :func:`cpp.activation_model`
      :type probs: pandas DataFrame
      :param sim: check_results table with simulation with or without drop-outs
      :type sim: pandas DataFrame
      :return:
         1) note about detected drop-outs (erroneously non-activated pools);
         2) list of the most possible peptides;
         3) list of all possible peptides given this pattern of pools activation.
      :rtype: list, list, list

      .. code-block:: python

         >>> note, most, possible = cpp.peptide_probabilities(sim, probs)
         >>> note
         No drop-outs were detected
         >>> most
         ['SSANNCTFEYVSQPFLM', 'CTFEYVSQPFLMDLEGK']
         >>> possible
         ['SSANNCTFEYVSQPFLM', 'CTFEYVSQPFLMDLEGK']

.. _simulation-section:

Play with the approach using simulated data (Optional)
-------------------------------------------------------

If you want to play with the approach with the generated data, you can use the following pipeline.

.. image:: simulation_pipeline.png

1. First, you need to determine the parameters for pooling scheme.

   * how many peptides? (len_lst)

   * how many pools? (n_pools)

   * what is peptide occurrence, i.e. to how many pools one peptide would be added? (iters)

   * what would be the length of the peptide? (pep_length)

   * what is the length of the shift between two overlapping peptides? (shift)

   * what is the length of the expected epitope (ep_length, we recommend 8)

2. Then, you can use these parameters to generate peptides. First, you would need to generate a random sequence, and then you could generate peptides using sliding window approach.
   
   .. code-block:: python

      >>> len_lst = 100
      >>> n_pools = 12
      >>> iters = 4
      >>> pep_length = 17
      >>> shift = 5
      >>> ep_length = 8

      >>> sequence = cpp.random_amino_acid_sequence(shift*len_lst + (100-shift*len_lst%100))
      >>> sequence
         'EMKFLDQSQLGYVHPKWHHGTEMDEWSRSNSAYGKHQEATRLCSQWWVKTYMPTDPCWMLRYTNCCAMVPRYADFCMRDYRYAYIYFVNWNHECSDVIMETCCFALGKKLSTPTCTPGCVTVIYECKSEFEVGWPPHIIEGSAEFYAVACFVTRFMCPQTKANLLKIIISFHLHHYGQAEQICYKNEIPCCAMKFFDHREGLESNCLTCMQWPCNKSLFDPFPVMYRFSMAGNQGEPPCGYAVTMNARCTMGRWQKFRCEFKGCFYHNINVYTGCETMHECQIPVPMVHQTTLLYPCNVRSKDIDPCDWSYLEDDKERGWCGKFQMGSQIFRKFTPPPWTNRGWNHMDDTEARHRWCLTWKFTLDEPAEDTCILWIHSVYLWVVCMQGTAMSMRMVSFTLLCFMRAPPCEVMHYCDPQQTRDEELPMVGYITEELKSMFTSSSWPGSQSPGWGTWDLSIKRHSVKVPDMINPTHVVKPTKCICNQSLGWTFSEIDMYARHDIQKRWKCPIWNGQFRYEVIHSKQNPFQNSDEQPT'

      ## Then with this sequence you can generate peptides
      >>> lst_all = cpp.peptide_generation(sequence, pep_length, shift)
      >>> lst = lst_all[:len_lst]

3. Then you can finally generate the pooling scheme.

   .. code-block:: python

      >>> b, lines = cdp.bba(m=n_pools, r=iters, n=len_lst)
      >>> pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=n_pools)
      >>> check_results = cpp.run_experiment(lst=lst, peptide_address=peptide_address, ep_length=ep_length, pools=pools, iters=iters, n_pools=n_pools, regime='without dropouts')

4. Then you need to select a cognate epitope to later check whether the model can recover it. You can do it manually if you particularly like some of them. But also you can do that randomly.

   .. code-block:: python

      >>> cognate = check_results.sample(1)['Epitope'][0]
      >>> check_results['Cognate'] = False
      >>> check_results.loc[check_results['Epitope'] == cognate, 'Cognate'] = True   
      >>> print(list(set(check_results['Peptide'][check_results['Epitope'] == cognate])))
      ['YCNQNWDWDMCEVVCGR', 'WDWDMCEVVCGRDFCHC']

   Also, you would need to find the pools which would be activated given this epitope is cognate.

   .. code-block:: python

      >>> inds_p_check = check_results[check_results['Cognate'] == True]['Act Pools'].values[0]

      >>> inds_p_check = [int(x) for x in inds_p_check[1:-1].split(', ')]
      >>> inds_n_check = []
      >>> for item in range(n_pools):
            if item not in inds_p_check:
               inds_n_check.append(item)
      >>> print(inds_p_check)
      [5, 6, 9, 10, 11]
      >>> print(inds_n_check)
      [0, 1, 2, 3, 4, 7, 8]

5. Then you can simulate activation signal. For that, you would need to determine paratemers of the model.

   Plate notation for the simulation model.

   .. image:: model_simulation.png

   * mu_n - mu of the negative distribution (distribution of signal of non-activated pools), ranges from 0 to 100.

   * sigma_n - sigma of the negative distribution, ranges from 0 to 100.

   * mu_off - mu of the offset which will be used to obtain positive distribution (distribution of signal of activated pools) from the negative distribution, ranges from 0 to 100.

   * sigma_off - sigma of the offset which will be used to obtain positive distribution, ranges from 0 to 100.

   * r - number of replicas in the experiment

   * sigma_p_r - variance between replicas from positive distribution, ranges between 0 to 100.

   * sigma_n_r - variance between replicas from negative distribution, ranges between 0 to 100.

   * n_pools - number of pools

   * p_shape - number of activated pools in simulation, you can make it equal to the number of pools where cognate epitope is present, or you can make more / fewer to see how the algorithm responds to mistakes.

   * pl_shape - number of slightly activated pools in simulation corresponding to context-dependent activation. For simplicity, we recommend setting it to 0.

   * low_offset - the degree to which activation is decreased in pools from pl_shape, ranges from 0 to 1. We recommend setting it to 1, then it will not be applied.


   .. code-block:: python

      >>> mu_off = 60
      >>> sigma_off = 3
      >>> mu_n = 20
      >>> sigma_n = 3
      >>> r = 1
      >>> sigma_p_r = 3
      >>> sigma_n_r = 3
      >>> n_pools = 12
      >>> p_shape = len(inds_p_check)  
      >>> pl_shape = 0
      >>> low_offset = 1

   .. code-block:: python

      >>> p_results, pl_results, n_results, n_control, parameters = cpp.simulation(mu_off, sigma_off, mu_n, sigma_n, r, sigma_p_r, sigma_n_r, n_pools, p_shape, pl_shape, low_offset)
      >>> cells = pd.DataFrame(columns = ['Pool', 'Percentage'])
      >>> cells['Percentage'] = p_results + n_results
      >>> cells['Pool'] = inds_p_check*r + inds_n_check*r

   Cells is a DataFrame with the simulated data:

   .. code-block:: python

      >>> cells

   .. table::
      :widths: 10 10

      +------+------------+
      | Pool | Percentage |
      +======+============+
      | 5    | 14.554757  |
      +------+------------+
      | 6    | 14.818329  |
      +------+------------+
      | 9    | 14.846125  |
      +------+------------+
      | 10   | 14.536968  |
      +------+------------+
      | 11   | 15.311202  |
      +------+------------+
      | 0    | 4.544784   |
      +------+------------+
      | 1    | 4.422958   |
      +------+------------+
      | 2    | 4.514103   |
      +------+------------+
      | 3    | 4.458392   |
      +------+------------+
      | 4    | 4.575509   |
      +------+------------+
      | 7    | 5.791510   |
      +------+------------+
      | 8    | 5.334201   |
      +------+------------+

6. Then you can use this table to check the algorithm.

   .. code-block:: python

      >>> inds = list(cells['Pool'])
      >>> obs = list(cells['Percentage'])
      >>> fig, probs = cpp.activation_model(obs, n_pools, inds)
      >>> peptide_probs = cpp.peptide_probabilities(check_results, probs)
      >>> cpp.results_analysis(peptide_probs, probs, check_results)
      ('No drop-outs were detected',
      ['YCNQNWDWDMCEVVCGR', 'WDWDMCEVVCGRDFCHC'],
      ['YCNQNWDWDMCEVVCGR', 'WDWDMCEVVCGRDFCHC'])

   Now you can compare recovered cognate peptides with ones you chose:

   * ['YCNQNWDWDMCEVVCGR', 'WDWDMCEVVCGRDFCHC'] - you chose
   
   * ['YCNQNWDWDMCEVVCGR', 'WDWDMCEVVCGRDFCHC'] - were recovered by the model from the simulated activation data

7. You can play with different parameters to check how well the approach works. For example, you can decrease the offset for the positive distribution, to check how different should be activated and non-activated pools to yield correct results.

.. _occurrence-section:

Peptide occurrence search
------------------------------

.. function:: cpp.factorial(num) -> int

      :param num: number
      :type n: int
      :return: factorial of the num
      :rtype: int

      .. code-block:: python

         >>> cpp.factorial(10)
         3628800

.. function:: cpp.combination(n, k) -> int

      :param n: set length
      :type n: int
      :return: how many items are selected from the set
      :rtype: int

      .. code-block:: python

         >>> cpp.combination(10, 3)
         120

.. function:: cpp.find_possible_k_values (n, l) -> list

      :param n: number of pools
      :type n: int
      :param l: number of peptides
      :type l: int
      :return: list with possible peptide occurrences given number of pools and number of peptides.
      :rtype: Counter object, dictionary

      .. code-block:: python

         >>> cpp.find_possible_k_values(12, 250)
         [4, 5, 6, 7, 8]

.. _overlap-section:

Peptide overlap
--------------------

.. function:: cpp.peptide_generation(protein, peptide_length, peptide_shift, protein_end=False) -> list
      :noindex:

      :param protein: a single protein sequence (string) or a list of protein sequences
      :type protein: str or list of str

      :param peptide_length: length of each generated peptide
      :type peptide_length: int

      :param peptide_shift: number of positions to shift between consecutive peptides (i.e., peptide_length - overlap)
      :type peptide_shift: int

      :param protein_end: whether to include trailing peptide if protein ends with a short fragment
      :type protein_end: bool, default is False

      :return: list of generated peptide sequences
      :rtype: list of str

      .. code-block:: python

         >>> peptides = cpp.peptide_generation("MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE", 9, 4)
         >>> peptides[:3]
         ['MKWVTFISL', 'TFISLLFLF', 'LLFLFSSAY']

      .. note::
         - If the input is a list of proteins, the peptides will be generated for each individually and concatenated.
         - If protein_end is True, peptides near the C-terminus will be padded by upstream sequence if shorter than expected.


.. function:: cpp.string_overlap(str1, str2): -> int

      :param str1: peptide
      :type str1: string
      :param str2: peptide
      :type str2: string
      :return: overlap length between two peptides
      :rtype: int

      .. code-block:: python

         >>> cpp.string_overlap('ASDFGHJKTYUIO', 'GHJKTYUIOTYUI')
         9

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

.. function:: cpp.bad_address_predictor(all_ns): -> list

      .. tip:: Initially it is designed for address arrangement produced by :func:`cpp.gc_to_address`. But keep in mind that produced arrangement might be imbalanced.

      :param all_ns: address arrangement
      :type all_ns: list
      :return: address arrangement without addresses with the same unions. The function searches for three consecutive addresses with the same union and removes the middle one.
      :rtype: list

      .. code-block:: python

         >>> cpp.bad_address_predictor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 2, 6], [0, 1, 3, 6], [0, 1, 3, 5], [0, 1, 3, 4]])
         [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 2, 6], [0, 1, 3, 6], [0, 1, 3, 5], [0, 1, 3, 4]]

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

.. function:: cpp.pools_activation(pools, epitope): -> list

      :param pools: pools, produced by :func:`cpp.pooling`
      :type pools: dictionary
      :param epitope: epitope present in one or several tested peptides
      :type epitope: string
      :return: pool indices where the epitope is present
      :rtype: list

      .. code-block:: python

         >>> cpp.pools_activation(pools, 'LGVYYHKN')
         [0, 3, 8, 9, 11]

.. function:: cpp.epitope_pools_activation(peptide_address, lst, ep_length): -> dictionary

      :param peptide_address: peptide addresses, produced by :func:`cpp.pooling`
      :type peptide_address: dictionary
      :param lst: ordered list of peptides
      :type lst: list
      :param ep_length: expected epitope length
      :type ep_length: ep
      :return: activated pools for every possible epitope of expected length from entered peptides
      :rtype: dictionary

      .. code-block:: python

         >>> cpp.epitope_pools_activation(peptide_address, lst, 8)
         {'[0, 1, 2, 3]': ['MFVFLVLL', 'FVFLVLLP', 'VFLVLLPL', 'FLVLLPLV', 'LVLLPLVS'], '[0, 1, 2, 3, 9]': ['VLLPLVSS', 'LLPLVSSQ', 'LPLVSSQC', 'PLVSSQCV', 'LVSSQCVN'], '[0, 1, 3, 9, 11]': ['VSSQCVNL', 'SSQCVNLT', ...], ... }

.. function:: cpp.peptide_search(lst, act_profile, act_pools, iters, n_pools, regime): -> list, list

      :param lst: ordered list of peptides
      :type lst: list
      :param act_profile: activated pools for every possible epitope of expected length from entered peptides, produced by :func:`cpp.epitope_pools_activation`
      :type act_profile: dictionary
      :param act_pools: activated pools
      :type act_pools: list
      :param iters: peptide occurrence
      :type iters: int
      :param n_pools: number of pools
      :type n_pools: int
      :param regime: regime of simulation, with or without drop-outs
      :type regime: "with dropouts" or "without dropouts"
      :return: possible peptides and possible epitopes given such activated pools
      :rtype: list, list

      .. code-block:: python

         >>> cpp.peptide_search(lst, act_profile, [0, 3, 8, 9, 11], 4, 12, 'without dropouts')
         (['CNDPFLGVYYHKNNKSW', 'LGVYYHKNNKSWMESEF'], ['LGVYYHKN', 'GVYYHKNN', 'VYYHKNNK', 'YYHKNNKS', 'YHKNNKSW'])
         >>> cpp.peptide_search(lst, act_profile, [0, 3, 8, 11], iters, n_pools, 'with dropouts')
         (['CNDPFLGVYYHKNNKSW', 'LLKYNENGTITDAVDCA', 'LGVYYHKNNKSWMESEF', 'QPRTFLLKYNENGTITD'], ['YNENGTIT', 'LKYNENGT', 'YHKNNKSW', 'KYNENGTI', 'YYHKNNKS', 'LGVYYHKN', 'VYYHKNNK', 'NENGTITD', 'LLKYNENG', 'GVYYHKNN'])

.. function:: cpp.run_experiment(lst, peptide_address, ep_length, pools, iters, n_pools, regime) -> pandas DataFrame

      .. note:: Simulation may take several minutes, especially upon "with drop-outs" regime.

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

.. function:: cpp.stl_generator(rows, cols, length, width, thickness, hole_radius, x_offset, y_offset, well_spacing, coordinates): -> Mesh object

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
      :param coordinates: coordinates of holes, in tuples in list
      :type coordinates: list
      :return: punch cards with holes based in entered coordinates
      :rtype: Mesh object

      .. code-block:: python

         >>> cpp.stl_generator(rows = 16, cols = 24, length = 122.10, width = 79.97, thickness = 1.5, hole_radius = 4.0 / 2, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5, [(1, 1), (2, 2), (1, 2)])
         Mesh object

.. function:: cpp.pools_stl(peptides_table, pools, rows = 16, cols = 24, length = 122.10, width = 79.97, thickness = 1.5, hole_radius = 4.0 / 2, x_offset = 9.05, y_offset = 6.20, well_spacing = 4.5) -> dictionary

      .. note:: Rendering of 3D models will take some time.

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

      Generated STL file you can check using OpenSCAD:
      
      .. image:: pools_stl.png
         :width: 400px
         :height: 200px

.. function:: cpp.zip_meshes_export(meshes_list) -> None

      :param meshes_list: dictionary with Mesh objects, generated by :func:`cpp.pools_stl`
      :type meshes_list: dictionary
      :return: export Mesh objects as STL files in .zip archive.
      :rtype: None

      .. code-block:: python

         >>> cpp.zip_meshes_export(meshes_list)

.. function:: cpp.zip_meshes(meshes_list): -> BytesIO object

      :param meshes_list: dictionary with Mesh objects, generated by :func:`cpp.pools_stl`
      :type meshes_list: dictionary
      :return: zip archive with generated STL files in BytesIO format (suitable for emails)
      :rtype: BytesIO

      .. code-block:: python

         >>> cpp.zip_meshes(meshes_list)
         <_io.BytesIO at 0x1d42a1440>

.. _interpretation:

Results interpretation with a Bayesian mixture model
------------------------------------------------------------

.. function:: cpp.activation_model(obs, n_pools, inds, neg_control=None, neg_share=None, cores=1) -> model, fig, pandas DataFrame, list, InferenceData, list
      :noindex:

      .. note:: Fitting might take several minutes.

      :param obs: list with observed values
      :type obs: list
      :param n_pools: number of pools
      :type n_pools: int
      :param inds: list with indices for observed values
      :type inds: list
      :param neg_control: optional list with negative control values; if not provided, it is estimated from obs
      :type neg_control: list or None
      :param neg_share: expected share of negative pools (between 0 and 1); default is 0.5
      :type neg_share: float or None
      :param cores: number of CPU cores to use for MCMC sampling
      :type cores: int
      :return:
         1) model -- PyMC model object used for fitting  
         2) fig -- posterior predictive KDE and observed data KDE (ArviZ)
         3) probs -- probability for each pool of being drawn from a distribution of activated or non-activated pools
         4) neg_control -- normalized control values used in model
         5) idata_alt -- full posterior sampling trace (InferenceData object)
         6) [p_mean, n_mean] -- posterior mean of the offset and baseline (negative) component
      :rtype: model, figure, pandas DataFrame, list, arviz.InferenceData, list

      .. code-block:: python

         >>> model, fig, probs, neg_control, trace, [p_mean, n_mean] = cpp.activation_model(obs, 12, inds)

      .. image:: model_fit.png

      .. code-block:: python

         >>> probs

      .. table::
         :widths: 10 10

         +------+---------+
         | Pool | assign  |
         +======+=========+
         | 0    | 0.99900 |
         +------+---------+
         | 1    | 1.00000 |
         +------+---------+
         | 2    | 0.00025 |
         +------+---------+
         | 3    | 0.36475 |
         +------+---------+
         | 4    | 0.00025 |
         +------+---------+
         | 5    | 0.00000 |
         +------+---------+
         | 6    | 1.00000 |
         +------+---------+
         | 7    | 1.00000 |
         +------+---------+
         | 8    | 0.99975 |
         +------+---------+
         | 9    | 0.99975 |
         +------+---------+
         | 10   | 0.00000 |
         +------+---------+
         | 11   | 0.99975 |
         +------+---------+

.. function:: cpp.peptide_probabilities(sim, probs) -> pandas DataFrame

      :param sim: check_results table with simulation with or without drop-outs
      :type sim: pandas DataFrame
      :param probs: DataFrame with probabilities produced by :func:`cpp.activation_model`
      :type probs: pandas DataFrame
      :return: peptide_probs -- probabilitity for each peptide to cause such a pattern of activation
      :rtype: pandas DataFrame

      .. code-block:: python

         >>> peptide_probs = cpp.peptide_probabilities(sim, probs)

.. function:: cpp.results_analysis(peptide_probs, probs, sim) -> list, list, list

      :param peptide_probs: DataFrame with probabilities for each peptide produced by :func:`cpp.peptide_probabilities`
      :type peptide_probs: pandas DataFrame
      :param probs: DataFrame with probabilities produced by :func:`cpp.activation_model`
      :type probs: pandas DataFrame
      :param sim: check_results table with simulation with or without drop-outs
      :type sim: pandas DataFrame
      :return:
         1) note about detected drop-outs (erroneously non-activated pools);
         2) list of the most possible peptides;
         3) list of all possible peptides given this pattern of pools activation.
      :rtype: list, list, list

      .. code-block:: python

         >>> note, most, possible = cpp.peptide_probabilities(sim, probs)
         >>> note
         No drop-outs were detected
         >>> most
         ['SSANNCTFEYVSQPFLM', 'CTFEYVSQPFLMDLEGK']
         >>> possible
         ['SSANNCTFEYVSQPFLM', 'CTFEYVSQPFLMDLEGK']

Data simulation with Bayesian mixture model
-----------------------------------------------

.. function:: cpp.random_amino_acid_sequence(length) -> str

      :param length: length of the random amino acid sequence from which peptides would be generated, calculate how long it should be for your number of peptides
      :type length: int
      :return: generated amino acid sequence of determined length
      :rtype: str

      .. code-block:: python

         >>> sequence = cpp.random_amino_acid_sequence(shift*len_lst + (100-shift*len_lst%100))
         >>> sequence
         'EMKFLDQSQLGYVHPKWHHGTEMDEWSRSNSAYGKHQEATRLCSQWWVKTYMPTDPCWMLRYTNCCAMVPRYADFCMRDYRYAYIYFVNWNHECSDVIMETCCFALGKKLSTPTCTPGCVTVIYECKSEFEVGWPPHIIEGSAEFYAVACFVTRFMCPQTKANLLKIIISFHLHHYGQAEQICYKNEIPCCAMKFFDHREGLESNCLTCMQWPCNKSLFDPFPVMYRFSMAGNQGEPPCGYAVTMNARCTMGRWQKFRCEFKGCFYHNINVYTGCETMHECQIPVPMVHQTTLLYPCNVRSKDIDPCDWSYLEDDKERGWCGKFQMGSQIFRKFTPPPWTNRGWNHMDDTEARHRWCLTWKFTLDEPAEDTCILWIHSVYLWVVCMQGTAMSMRMVSFTLLCFMRAPPCEVMHYCDPQQTRDEELPMVGYITEELKSMFTSSSWPGSQSPGWGTWDLSIKRHSVKVPDMINPTHVVKPTKCICNQSLGWTFSEIDMYARHDIQKRWKCPIWNGQFRYEVIHSKQNPFQNSDEQPT'

.. function:: cpp.simulation(mu_off, sigma_off, mu_n, sigma_n, r, sigma_p_r, sigma_n_r, n_pools, p_shape, pl_shape, low_offset, cores=1) -> list, list, list, list, list
      :noindex:

      .. note:: Generation might take several minutes.

      :param mu_off: mu of the Truncated Normal distribution for the offset.
      :type mu_off: float, from 0 to 100

      :param sigma_off: sigma of the Truncated Normal distribution for the offset.
      :type sigma_off: float, from 0 to 100

      :param mu_n: mu of the Truncated Normal distribution for the negative (non-activated) signal source.
      :type mu_n: float, from 0 to 100

      :param sigma_n: sigma of the Truncated Normal distribution for the negative signal source.
      :type sigma_n: float, from 0 to 100

      :param r: number of replicas for each pool
      :type r: int

      :param sigma_p_r: sigma of measurement error for positive and low-positive pools (replicate variability)
      :type sigma_p_r: float, from 0 to 100

      :param sigma_n_r: sigma of measurement error for negative pools (replicate variability)
      :type sigma_n_r: float, from 0 to 100

      :param n_pools: total number of pools in the experiment
      :type n_pools: int

      :param p_shape: number of strongly activated pools
      :type p_shape: int

      :param pl_shape: number of low-activated (weak signal) pools
      :type pl_shape: int

      :param low_offset: scalar between 0 and 1 to reduce signal intensity for low-activated pools
      :type low_offset: float

      :param cores: number of CPU cores to use for MCMC sampling
      :type cores: int

      :return:
         1) p_results -- simulated signal values (mean) for activated pools  
         2) pl_results -- simulated signal values for low-activated pools  
         3) n_results -- simulated signal values for non-activated pools  
         4) n_control -- simulated values for negative control  
         5) [p_mean, n_mean] -- posterior means of offset and baseline signal
      :rtype: list, list, list, list, list

      .. code-block:: python

         >>> p, pl, n, control, [offset_mean, neg_mean] = cpp.simulation(
         ...     mu_off=10, sigma_off=0.5, mu_n=5, sigma_n=1, r=1,
         ...     sigma_p_r=0.2, sigma_n_r=0.3, n_pools=16, p_shape=4, pl_shape=2, low_offset=0.5
         ... )

         >>> p
         [15.1, 14.8, 15.0, 15.2]
         >>> pl
         [7.2, 7.4]
         >>> n
         [5.1, 5.2, 5.0, 5.3, 5.1, 5.0, 5.2, 5.3, 5.0, 5.1]
         >>> control
         [4.9, 5.0, 5.1]


