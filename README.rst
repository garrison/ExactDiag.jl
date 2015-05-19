=========
ExactDiag
=========

Very much a work in progress.

See also: IndexedArrays, LinTables.  And also Bravais.

Available model systems
=======================

Spin 1/2
--------

Spin up = 1, spin down = 0

Hard-core (?) bosons
--------------------

Presence of particle = 1, absence = 0.

Fermionic Hubbard models
------------------------

0,1,2,3

Diagonalization of translationally invariant system
===================================================

.. todo:: old assumptions: PBC, and no fermions involved.

Typically, our goal is to find one or more eigenstates of the Hamiltonian :math:`\hat{H}`.  When :math:`\hat{H}` is translation invariant, we can change our basis such that :math:`\hat{H}` is diagonal in each momentum sector.  We can take advantage of this by then diagonalizing each sector independently, or diagonalizing just the sector(s) we are interested in.  This uses both less processor time and less memory for a given system size.

So how exactly do we diagonalize each momentum sector separately?

Consider a projection operator

.. math::
   \hat{P}_\mathbf{k} \equiv \frac{1}{N} \sum_\mathbf{r} e^{i\mathbf{k}\cdot \mathbf{r}} \prod_{i=1}^{d} \hat{T}_i^{r_i}

where :math:`\mathbf{r_i}` is defined by :math:`\mathbf{r} = \sum_{i=1}^d r_i \mathbf{a}_i` (where :math:`\mathrm{a}_i` are the primitive vectors of the lattice), :math:`\hat{T}_i` is the unit translation operator in the :math:`i`'th direction of the lattice, and :math:`\mathbf{k}` is some allowed momenta of the system.  (In a one dimensional spin-1/2 system of length :math:`L` with PBC, the translation operator is defined such that :math:`\hat{T}_1 \vert \sigma_1 \cdots \sigma_{L-1} \sigma_L \rangle = \vert \sigma_L \sigma_1 \cdots \sigma_{L-1} \rangle`, and :math:`k= \frac{2\pi k_\mathrm{idx}}{L}` where :math:`k_\mathrm{idx} \in \mathbb{Z}_L`.)

Since :math:`[\hat{H}, \hat{T}] = 0`, it follows that :math:`[\hat{H}, \hat{P}_k] = 0`.  It can also be shown that :math:`\hat{P}_k^\dagger = \hat{P}_k` and :math:`\hat{P}_k^2 = \hat{P}_k`.  In other words, :math:`\hat{P}_k` is a Hermitian projection operator that commutes with the Hamiltonian.

We can use this operator to project an arbitrary "representative" state :math:`\vert r \rangle` to a momentum state :math:`\hat{P}_k \vert r \rangle`.  If :math:`\hat{P}_k \vert r \rangle = 0`, there is no such state at momentum :math:`k` represented by :math:`\vert r \rangle`.  However, if :math:`\hat{P}_k \vert r \rangle \ne 0`, we can define a normalized state

.. math::
   \vert r_k \rangle \equiv \frac{1}{\mathcal{N}_{r_k}} \hat{P}_k \vert r \rangle

where :math:`\mathcal{N}_{r_k} = \sqrt{\langle r \vert \hat{P}_k \vert r \rangle}` such that :math:`\langle r_k \vert r_k \rangle = 1`.  Note that :math:`\vert r_k \rangle` is an eigenstate of :math:`\hat{T}` with eigenvalue :math:`e^{-ik}`.

For a concrete example, consider a 1D system with :math:`L=4` and PBC.  The representative state :math:`\vert \uparrow \downarrow \downarrow \downarrow \rangle` can exist at any available momentum in the system.  For instance, at :math:`k=\pi / 2`, the corresponding momentum state becomes

.. math::
   \vert \uparrow \downarrow \downarrow \downarrow _{\pi/2} \rangle
   \equiv \frac{1}{2} \left[
   \vert \uparrow \downarrow \downarrow \downarrow \rangle
   + i \vert \downarrow \uparrow \downarrow \downarrow \rangle
   - \vert \downarrow \downarrow \uparrow \downarrow \rangle
   - i \vert \downarrow \downarrow \downarrow \uparrow \rangle \right]

Now consider instead the representative state :math:`\vert \uparrow \downarrow \uparrow \downarrow \rangle`.  There is no such state at momentum :math:`\pi/2`, since :math:`\hat{P}_{\pi/2} \vert \uparrow \downarrow \uparrow \downarrow \rangle = 0`.  However, there are states at momenta :math:`0` and :math:`\pi`.  For instance,

.. math::
   \vert \uparrow \downarrow \uparrow \downarrow _\pi \rangle
   \equiv \frac{1}{\sqrt{2}} \left[
   \vert \uparrow \downarrow \uparrow \downarrow \rangle
   - \vert \downarrow \uparrow \downarrow \uparrow \rangle
   \right]

With this in mind we generally act as follows.  We choose a unique representative state for each class of states that are connected to each other by translation.  Then, given a momentum k, we go through each representative state and calculate its normalization :math:`\mathcal{N}_{r_k}`.  We consider each state :math:`\vert r_k \rangle` where :math:`\mathcal{N}_{r_k} \ne 0` to be part of our basis in this momentum sector.  We can then evaluate the matrix elements, given by

.. math::
   \langle r_k^\prime \vert \hat{H} \vert r_k \rangle
   = \frac{1}{\mathcal{N}_{r_k^\prime}\mathcal{N}_{r_k}} \langle r^\prime \vert \hat{P}_k \hat{H} \hat{P}_k \vert r \rangle
   = \frac{1}{\mathcal{N}_{r_k^\prime}\mathcal{N}_{r_k}} \langle r^\prime \vert \hat{P}_k \hat{H} \vert r \rangle

where the last part uses :math:`[\hat{H}, \hat{P}_k] = 0` and :math:`\hat{P}_k^2 = \hat{P}_k`.

We diagonalize :math:`\hat{H}` in this basis given by :math:`\vert r_k \rangle`; that is, we diagonalize the matrix given by matrix elements :math:`\langle r_k^\prime \vert \hat{H} \vert r_k \rangle`.  Given the eigenstates in this basis, we can recover our eigenstates in the original (position space) basis by evaluating :math:`\langle s \vert r_\mathbf{k} \rangle` using the definition of :math:`\vert r_k \rangle` above.

NOTE: we want to be able to do both unitaries; transforming to momentum basis, and back from it to position basis.

Other code notes
================

``IndexedArrays`` is used to assign each state (e.g. ``[1, 0, 1, 1, 0]`` in Julia land) to an index, and vice versa.  Given :math:`M` unique states, the indices will range from :math:`1` to :math:`M`.
