Installation and Dependencies
=============================

The current release of StraKLIP is not stable enough for a straightforward install with conda / pip. At this stage
it is recommended that you clone the git repository directory for installation:

::

	git clone https://github.com/strampelligiovanni/StraKLIP.git

If you would like a specific branch:

::

	git clone https://github.com/strampelligiovanni/StraKLIP.git@branch

From here, it is **highly** recommended that you create a unique anaconda environment to hold all of the StraKLIP
dependencies. StraKLIP is not currently compatible with python 3.12

::

	conda create -n straklip python=3.11
	conda activate straklip

With the anaconda environment created, move to the cloned directory and install most of the dependencies:

::

	cd where/you/installed/the/git/repo
	pip install -r requirements.txt
	pip install -e .