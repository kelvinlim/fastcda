# fastcda

fastcda is a package for performing causal discovery analysis.

The primary driver of this project is to create a package that can be installed quickly with minimal friction, support multiple platforms (Linux, Windows, macOS) and have fast execution and the ability to handle large datasets efficiently.  Consequently, the goal is to have the core causal search algorithms written in C with the other "glue" components written in Python.

During the initial phase, we use jpype to call methods in the Tetrad java program from Carnegie Mellon University (https://github.com/cmu-phil/tetrad).  This will also facilitate comparison between algorithms. The default Tetrad version being used in 7.6.3.  This corresponds to causal_cmd 1.12.0.

The code has been designed and tested to run on Windows11, macOS Sequoia and Ubuntu 22.04.  It should run on other versions of these platforms.

For a simple sample usage example, try out the fastcda_demo_short.ipynb file in the github repository. This will run nicely within vscode.

## Usage

### 1. Preliminaries

You will need a JDK21 or higher version which can be downloaded from here: https://www.oracle.com/java/technologies/downloads/#java21

You will also need the graphviz package which can be downloaded from here: https://graphviz.org/download/

On initialization, FastCDA will check that it can find the
needed java version and the graphviz dot program. If it is
unable to find the necessary programs, it will complain and
exit.

If you have installed the packages in non standard locations, you
can use a yaml configuration file to specify the locations.

For linux/macos, the configuration file  should
be in the home directory, e.g.  ~/.fastcda.yaml.  For Windows,
the file should be placed in the Users home directory AppData\Local

C:\Users\<YourUser>\AppData\Local

Here is a sample yaml configuration file.

```
# configuration file for FastCDA
# This file is used to set environment variables and paths for the FastCDA application

# JAVA_HOME is the path to the Java Development Kit (JDK)
# Ensure that the JDK version is compatible with FastCDA
JAVA_HOME: /Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home

# graphviz is a graph visualization software used by FastCDA
# Ensure that the Graphviz binaries are installed and the path is correct
GRAPHVIZ_BIN: /opt/homebrew/bin
```

### 2. Create a python virtual environment

```
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Then install the necessary packages using pip
pip install fastcda

```

### 3. Sample usage

a. Load the packages and create an instance of FastCDA

```
from fastcda import FastCDA
from dgraph_flex import DgraphFlex
import semopy
import pprint as pp

# create  an instance of FastCDA
fc = FastCDA()
```

b. Read in the built in sample ema dataset

```
# read in the sample data set in to a dataframe
df = fc.getEMAData()

# add the lags, with a suffix of '_lag'
df_lag = fc.add_lag_columns(df, lag_stub='_lag')

# standardize the data
df_lag_std = fc.standardize_df_cols(df_lag)

```

c. Create the prior knowledge content

```
# Create the knowledge prior content for temporal
# order. The lag variables can only be parents of the non
# lag variables
knowledge = {'addtemporal': {
                            0: ['alcohol_bev_lag',
                                'TIB_lag',
                                'TST_lag',
                                'PANAS_PA_lag',
                                'PANAS_NA_lag',
                                'worry_scale_lag',
                                'PHQ9_lag'],
                            1: ['alcohol_bev',
                                'TIB',
                                'TST',
                                'PANAS_PA',
                                'PANAS_NA',
                                'worry_scale',
                                'PHQ9']
                            }
            }
```

d. Run the search

```
# run model with run_model_search
result, graph = fc.run_model_search(df_lag_std, 
                             model = 'gfci',
                             score={'sem_bic': {'penalty_discount': 1.0}},
                             test={"fisher_z": {"alpha": .01}},
                             knowledge=knowledge
                             )
```

e. Show the causal graph

```
graph.show_graph()
```

![Example Graph](https://github.com/kelvinlim/fastcda/blob/main/assets/causal_graph_boston.png)
