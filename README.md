Python Coupled Neuron Explorer

This Python tool allows users to explore the dynamics that
are presented by coupled neurons in an interactive setting.

For UNIX like systems, it is reccomended that you follow
these steps to set up the system:

1. Make sure you have Python 3 installed on your system, type
   `$ python --version`
   to check the version, if you are running < 3, type:
   `$ python3 --version`
   On some systems, both Pythons are installed next to each
   other and Python 3 is accessed by the second command. If
   you do not have Python installed, use your operating
   system's package manager to install Python 3 (Homebrew on
   iOS, apt-get on Ubuntu, ect). Look online for instructions
   for installing Python 3 on your operating systme.
   
2. Create a virtual environment:
   Make sure that virtualenv is installed by running
   `$ which virtualenv`
   and installing with:
   `$ pip install virtualenv`
   if virtualenv was not installed on your system. Next,
   create a new virtual environment running Python 3 with:
   `$ virtualenv env -p python3`
   or, if your python is symlinked to 'python' then:
   `$ virtualenv env -p python`
   Activate your virtual environment by typing:
   `$ source env/bin/activate`

3. Install the dependancies:
   Run the following shell glob to install the required
   packages:
   `(env) $ pip install {matplotlib,scipy}`

4. Finally, run the explorer with the command:
   `(env) $ python run_explorer.py`

The application may hang on closure and need to be closed
from the command line.