#! /usr/bin/env python

from importlib.resources import files as pkg_resources_files # For Python 3.9+

import json
import os
import re
from pathlib import Path
import socket
import subprocess
from typing import Optional, List

import jpype
import jpype.imports

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import semopy
import yaml

from dgraph_flex import DgraphFlex

from sklearn.preprocessing import StandardScaler

__version_info__ = ('0', '1', '8')
__version__ = '.'.join(__version_info__)

version_history = \
"""
0.1.8 - add support for providing a custom Tetrad JAR file path
        to the FastCDA object. If not provided, the default JAR file is used.
0.1.7 - support min_java_version to 17
0.1.6 - change how configuration is done.  Eliminate the .tetradrc file.
        Use a fastcda.yaml file instead containing these two variables:
        
        JAVA_HOME: /Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home   
        GRAPHVIZ_BIN: /opt/homebrew/bin
        
        Only use the file if the default system paths are not sufficient.
        
0.1.5 - move to the FastCDA repository
0.1.4 - add verbose message option in object init
        run_model_search now automatically does the sem and creates the
        graph object with the results
0.1.3 - add checks of java_version and graphviz dot
0.1.2 - reworked run_model_search to use custom run_gfci from Bryan Andrews
0.1.1 - change startJVM to use jars in the package
0.1.0 - initial version  
"""
class FastCDA():
    
    def __init__(self, 
                 verbose: int = 1,
                 min_java_version: int = 17,
                 tetrad_jar_path: Optional[str] = None
                 ):
        
        """
        Initialize the FastCDA object

        Args:
        verbose - message verbosity, default is level 1
        min_java_version - minimum java version to accept
        tetrad_jar_path - path to the Tetrad JAR file, if None, uses the default JAR in the package
        
        """
        
        # default verbose level is 1, 2 is more chatty
        # output about what is happening
        self.verbose = verbose
        
        # check if a configuration file exists, then uses it
        res = self.loadPaths()
        
        self.min_java_version = min_java_version
        
        # check if we have graphviz dot on path
        graphviz_check, graphviz_version = self.check_graphviz_dot()
        
        exit_code = 0  # initialize exit code
        
        if not graphviz_check:
            print("Error: Graphviz 'dot' command is not found. Please install Graphviz and ensure it's in your PATH.")
            exit_code += 1  # Increment exit code if Graphviz is not found
            
        # check if we have correct Java version
        if not self.check_java_version(min_java_version=self.min_java_version):
            print(f"Error: Java version must be {self.min_java_version} or higher. Please update your Java installation.")
            exit_code += 1  # Increment exit code if Java version is not sufficient
        
        if exit_code > 0:
            # exit if any errors were found
            raise EnvironmentError(f"Exiting due to {exit_code} errors. Please resolve the issues and try again.")
        
        # only start if not already started
        if  not jpype.isJVMStarted():
            self.startJVM(tetrad_jar_path=tetrad_jar_path)
        else:
            if self.verbose >= 1: print("Java Virtual Machine already running...")
        pass
    
    def startJVM(self, 
                 jvm_args="-Xmx8g",
                 tetrad_jar_path: Optional[str] = None
                 ):
        
        """
        Start the JVM using the provided args
        
        Args:
        
        """
        # check if a custom JAR path is provided
        if tetrad_jar_path is not None:
            if not os.path.exists(tetrad_jar_path):
                raise ValueError(f"Provided JAR path {tetrad_jar_path} does not exist.")
            classpath = tetrad_jar_path
        else:
            # Use the default JAR file from the package resources
            # Determine the path to the JAR file dynamically if inside package
            jar_filename = "tetrad-gui-7.6.3-launch.jar"
            jar_resource = pkg_resources_files('fastcda.jars').joinpath(jar_filename)
            classpath = str(jar_resource) # This gives a Path object, convert to string for jpype

        if self.verbose > 1: print(f"Attempting to start JVM with classpath: {classpath}")
        res = jpype.startJVM(jvm_args, classpath=classpath)
        
        # make the classes available within the class
        self.util = jpype.JPackage("java.util")
        self.td = jpype.JPackage("edu.cmu.tetrad.data")
        self.tg = jpype.JPackage("edu.cmu.tetrad.graph")
        self.ts = jpype.JPackage("edu.cmu.tetrad.search")
        self.knowledge = self.td.Knowledge()
        
        self.lang = jpype.JPackage("java.lang")

    def getTetradVersion(self):
        """
        Get the version of Tetrad
        """
        
        from edu.cmu.tetrad.util import Version
        
        version = Version.currentViewableVersion().toString()
        return version

    def get_java_version(self):
        """
        
        Get the Java version installed on the system.

        typical string is
        
        'java version "21.0.6" 2025-01-21 LTS\nJava(TM) SE Runtime Environment (build 21.0.6+8-LTS-188)\nJava HotSpot(TM) 64-Bit Server VM (build 21.0.6+8-LTS-188, mixed mode, sharing)\n'
        
        Returns:
            _type_: _description_
        """
        try:
            # Execute the command. stderr=subprocess.PIPE is important because
            # java -version often prints to stderr.
            process = subprocess.run(['java', '-version'], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            
            # The output of 'java -version' is typically on stderr
            version_output = process.stderr
            return version_output
        except FileNotFoundError:
            return "Java is not found. Make sure it's installed and in your PATH."
        except subprocess.CalledProcessError as e:
            return f"Error executing command: {e}\nOutput: {e.stderr}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
    
    def check_java_version(self, min_java_version=21):
        """
        Check if the Java version is 21 or higher.
        
        Returns:
            bool: True if Java version is 21 or higher, False otherwise.
        """
        java_version = self.get_java_version()
        # Extract the version number from the string
        match = re.search(r'(\d+)\.(\d+)', java_version)
        if match:
            major_version = int(match.group(1))
            minor_version = int(match.group(2))
            return (major_version >= min_java_version) and minor_version >= 0
        return False 

    def check_graphviz_dot(self):
        """
        Checks if the Graphviz 'dot' command is installed and returns its version.

        sample output
        "dot - graphviz version 12.2.1 (20241206.2353)"
        
        Returns:
            tuple: (bool, str or None)
                - True if 'dot' is found, False otherwise.
                - The version string if found, None otherwise.
        """
        try:
            # Run 'dot -V' command to get the version
            # capture_output=True to get stdout and stderr
            # text=True to decode output as text
            result = subprocess.run(['dot', '-V'], capture_output=True, text=True, check=True)
            # The version is usually in stderr for 'dot -V'
            version_output = result.stderr.strip()
            
            # Parse the version string. It typically looks like:
            # "dot - graphviz version 2.47.1 (20210417.1919)"
            if "graphviz version" in version_output:
                version_start = version_output.find("graphviz version") + len("graphviz version")
                version_end = version_output.find(")", version_start)
                version = version_output[version_start:version_end].strip()
                return True, version
            else:
                return True, version_output # Fallback if format changes
        except FileNotFoundError:
            return False, None
        except subprocess.CalledProcessError as e:
            # 'dot' command was found but exited with an error
            # This might happen if there's an issue with the Graphviz installation
            print(f"Error running 'dot -V': {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return False, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False, None
    
    def loadPaths(self):
        """
        Checks if a configuration file exists in the home directory for macos/linux 
        and in %APPDATA% for windows.
        
        The yaml file should contain the following variables:
    
        JAVA_HOME and GRAPHVIZ_BIN
        
        For windows the file would be in the fastcda directory in the user's AppData folder.
        The file is named fastcda.yaml and 
        
        For macos/linux the file is in the home directory and is named .fastcda.yaml
        
        """
        
        # first lets check which os we are on
        if os.name == 'nt':  # Windows
            # get the AppData directory
            appdata_dir = os.getenv('LOCALAPPDATA')
            config_file = os.path.join(appdata_dir, 'fastcda', 'fastcda.yaml')
        else:  # macOS/Linux
            # get the home directory
            home_directory = Path.home()
            config_file = home_directory / '.fastcda.yaml'
            
        # check if the file exists
        if os.path.exists(config_file):
            # read in the yaml file
            try: 
                with open(config_file, 'r') as file:
                    config = yaml.safe_load(file)
            except:
                raise ValueError(f"Unable to read configuration file {config_file}. Please check the file format.")

            # check if JAVA_HOME exists in config
            if 'JAVA_HOME' in config:
                java_home = config['JAVA_HOME']
                if not os.path.exists(java_home):
                    raise ValueError(f"JAVA_HOME {java_home} does not exist. Please check the path.")
                
                # set the JAVA_HOME environment variable
                os.environ['JAVA_HOME'] = java_home
                
                # add the java bin directory to the PATH
                java_path = f"{java_home}/bin"
                current_path = os.environ.get('PATH')
                # add this to PATH
                os.environ['PATH'] = f"{current_path}{os.pathsep}{java_path}"

            # check if GRAPHVIZ_BIN exists in config
            if 'GRAPHVIZ_BIN' in config:
                graphviz_bin = config['GRAPHVIZ_BIN']
                if not os.path.exists(graphviz_bin):
                    raise ValueError(f"GRAPHVIZ_BIN {graphviz_bin} does not exist. Please check the path.")
                
                # add the graphviz bin directory to the PATH
                current_path = os.environ.get('PATH')
                # add to path
                os.environ['PATH'] = f"{current_path}{os.pathsep}{graphviz_bin}"
                

        
    def getEMAData(self) -> pd.DataFrame:
        """
        Returns a dataframe containing an EMA dataset
        
        Returns:
        pandas dataframe
        """
        
        csv_filename = "boston_data_raw.csv"
        csv_resource = pkg_resources_files('fastcda.data').joinpath(csv_filename)
        csv_path = str(csv_resource) # This gives a Path object, convert to string for jpype
        df = pd.read_csv(csv_path)
        return df
    
    def df_to_data(self, 
                   df: pd.DataFrame,
                   jitter: bool=False):
        """
        
        Load pandas dataframe into TetradSearch

        Args:
            df (pd.DataFrame): dataframe
            jitter (bool): flag to jitter data

        Returns:
            
        """
        cols = df.columns
        values = df.values
        n, p = df.shape

        # JITTER THE DATA; FEEL FREE TO REMOVE
        if jitter:
            values += 1e-3 * np.random.randn(n, p)

        variables = self.util.ArrayList()
        for col in cols:
            variables.add(self.td.ContinuousVariable(str(col)))

        databox = self.td.DoubleDataBox(n, p)
        for col, var in enumerate(values.T):
            for row, val in enumerate(var):
                databox.set(row, col, val)

        return self.td.BoxDataSet(databox, variables)

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """Read a CSV file and return a pandas DataFrame.
        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: pandas DataFrame
        """
        self.full_df = pd.read_csv(file_path)
        return self.full_df

    def add_lag_columns(self, df: pd.DataFrame, lag_stub='_') -> pd.DataFrame:
        """
        Lag the dataframe by shifting the columns by one row

        Args:
            df (pd.DataFrame): the dataframe to lag
            lag_stub (str): the string to append to the column names for the lagged variables

        Returns:
            pd.DataFrame: the lagged dataframe
        """
        
        # create a copy of the dataframe
        df_lag = df.copy()
        
        # create additional columns for the lagged variables, naming them  lagdrinks, lagsad, etc.
        cols_to_lag = df.columns.tolist()
        # shift by one row
        for col in cols_to_lag:
            df_lag[f'{col}{lag_stub}'] = df[col].shift(1)
        
        # drop the first row
        df_lag = df_lag.dropna()
        
        # reset index
        df_lag = df_lag.reset_index(drop=True)
        
        return df_lag

   
    def load_df_into_ts(self,df):
        """
        Loads a pandas DataFrame into the TetradSearch object.
        """
        self.data = tr.pandas_data_to_tetrad(df)
        return self.data
    
    def subsample_df(self, df: Optional[pd.DataFrame] = None,    
                    fraction: float = 0.9,
                    random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Randomly subsample the DataFrame to a fraction of rows
        Args:
            df - pandas DataFrame
            fraction - proportion of rows to keep, default 0.9
            random_state - random state for reproducibility
        Returns:
            df - pandas DataFrame
        """
        if df is None:
            if hasattr(self, 'full_df') and self.full_df is not None:
                df = self.full_df
            else:
                raise ValueError("DataFrame must be provided.")
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"fraction must be between 0 and 1")
        
        # Use the DataFrame's built-in sample method with the specified fraction
        # The `random_state` parameter ensures reproducibility if an integer is provided
        scrambled_df = df.sample(frac=fraction, random_state=random_state)

        # Step 2: Sort the sampled DataFrame by its original index
        # This restores the original relative order of the kept rows
        self.subsampled_df = scrambled_df.sort_index()
        
        return self.subsampled_df

    def standardize_df_cols(self, df, diag=False):
        """
        standardize the columns in the dataframe
        https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/
        
        * get the column names for the dataframe
        * convert the dataframe into  a numeric array
        * scale the data
        * convert array back to a df
        * add back the column names
        * set to the previous df
        """
        
        # describe original data - first two columns
        if diag:
            print(df.iloc[:,0:2].describe())
        # get column names
        colnames = df.columns
        # convert dataframe to array
        data = df.values
        # standardize the data
        std_data = StandardScaler().fit_transform(data)
        # convert array back to df, use original colnames
        newdf = pd.DataFrame(std_data, columns = colnames)
        # describe new data - first two columns
        if diag:
            print(newdf.iloc[:,0:2].describe())
        
        return newdf
    
    def create_permuted_dfs(self, df: pd.DataFrame, n_permutations: int, seed: int = None) -> List[pd.DataFrame]:
        """
        Generates multiple DataFrames, each with elements permutated within columns independently.

        Args:
            df: The input pandas DataFrame.
            n_permutations: The number of permutated DataFrames to generate.
            seed: An optional integer seed for the random number generator
                to ensure reproducibility of the sequence of permutations.
                If None, the permutations will be different each time.

        Returns:
            A list containing n pandas DataFrames, each being a column-wise
            permutation of the original DataFrame.
        """
        # Check if the input DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")

        # Check if n is a positive integer
        if not isinstance(n_permutations, int) or n_permutations <= 0:
            raise ValueError("The number of permutations (n) must be a positive integer.")

        # If a seed is provided, set it for reproducibility
        # This allows for consistent random permutations across different runs
        # If no seed is provided, the permutations will be different each time
        # Note: np.random.default_rng() is used to create a new random number generator instance
        # This is a more modern approach compared to np.random.seed() and allows for better control
        # over random number generation.
        # The seed is used to initialize the random number generator
        # This ensures that the same sequence of random numbers is generated each time
        # the same seed is used.
        # This is useful for debugging and testing purposes
        # If no seed is provided, the permutations will be different each time
        # the function is called.   
        # Initialize the random number generator once
        # If a seed is provided, the sequence of generated permutations will be reproducible
        rng = np.random.default_rng(seed)

        permutated_dfs = [] # List to store the resulting DataFrames

        for _ in range(n_permutations): # Generate n permutations
            # Create a fresh copy of the original DataFrame for each permutation
            df_permuted = df.copy()

            # Permutate each column independently using the same RNG state
            for col in df_permuted.columns:
                df_permuted[col] = rng.permutation(df_permuted[col].values)

            # Add the newly permutated DataFrame to the list
            permutated_dfs.append(df_permuted)

        return permutated_dfs

    
    def select_edges(self, edge_counts: dict, min_fraction: float) -> dict:
        """
        Select edges based on a fraction of the total edges
        Args:
            edges - dictionary with the edges and their counts
            min_fraction - mininum fraction of edges to select
        Returns:
            dict - dict of selected edges as keys and their fraction as values
        """

        selected_edges = {}  # holds the selected edges as strings e.g. 'A --> B' with fraction
        
        # create a dataframe with src, edge, dest and fraction, extract the src, edge, dest from each edge
        # and add to the dataframe
        edge_df = pd.DataFrame(edge_counts.items(), columns=['edge', 'fraction'])
        edge_df[['src', 'edge_type', 'dest']] = edge_df['edge'].str.split(' ', expand=True)
        # drop the edge column
        edge_df = edge_df.drop(columns=['edge'])
        
        # first lets process the edge_type of --> and o->
        # filter the df for edge_type of --> and o->
        directed_edge_df = edge_df[edge_df['edge_type'].isin(['-->', 'o->'])]
        # sort the directed edges by src, dest, and fraction (descending)
        directed_edge_df = directed_edge_df.sort_values(by=['src', 'dest', 'fraction'], ascending=[True, True, False])

        # get the rows where the src and dest only appear once in the df
        single_directed_edges = directed_edge_df.groupby(['src', 'dest']).filter(lambda x: len(x) == 1)
        
        # iterate over the rows and add them to the edge list if the fraction is >= min_fraction
        for index, row in single_directed_edges.iterrows():
            if row['fraction'] >= min_fraction:
                # create the edge string
                edge_str = f"{row['src']} {row['edge_type']} {row['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = row['fraction']
                pass
            
        # now lets process the directed edges that have multiple rows                   
        # keep the rows where the src and dest are the same across rows
        multiple_directed_edges = directed_edge_df.groupby(['src', 'dest']).filter(lambda x: len(x) > 1)
        
        # iterate over two rows, and if the sum of fraction is greater than min_fraction then keep the edge
        for i in range(0, len(multiple_directed_edges), 2):
            row_pairs = multiple_directed_edges.iloc[i:i+2]
            # get the sum of the fraction
            fraction_sum = row_pairs['fraction'].sum()
            # check if the sum is greater than min_fraction
            if fraction_sum >= min_fraction:
                # create the edge string
                # we use the first row to create the edge string since it has the highest fraction
                edge_str = f"{row_pairs.iloc[0]['src']} {row_pairs.iloc[0]['edge_type']} {row_pairs.iloc[0]['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = float(fraction_sum)
                pass
            pass
        
        # get the undirected edges
        undirected_edges_df = edge_df[~edge_df['edge_type'].isin(['-->', 'o->'])]

        # iterate over the df using iloc, if the src > dest then swap them in the row and update the df
        # this will make sure that the edges are in the same order
        # for example, A o-o B and B <-> A will be made adjacent to each other when sorted
        for i in undirected_edges_df.index:
            # Use .loc to access the specific row and columns directly
            # This ensures you are working with the original DataFrame
            # and prevents the SettingWithCopyWarning
            if undirected_edges_df.loc[i, 'src'] > undirected_edges_df.loc[i, 'dest']:
                # Swap the values directly in the DataFrame using .loc
                # We use tuple assignment for a clean swap
                undirected_edges_df.loc[i, 'src'], undirected_edges_df.loc[i, 'dest'] = \
                undirected_edges_df.loc[i, 'dest'], undirected_edges_df.loc[i, 'src']
            
            
            # row = undirected_edges_df.iloc[i]
            # if row['src'] > row['dest']:
            #     # swap the src and dest
            #     temp = row['src']
            #     row['src'] = row['dest']
            #     row['dest'] = temp
            #     # update the row in the df using iloc
            #     undirected_edges_df.iloc[i] = row
            pass
    
        # sort the undirected edges by src, dest, and fraction (descending)
        undirected_edges_df = undirected_edges_df.sort_values(by=['src', 'dest', 'fraction'], ascending=[True, True, False])
        # get the undirected edges that have single rows
        single_undirected_edge_df = undirected_edges_df.groupby(['src', 'dest']).filter(lambda x: len(x) == 1)
        
        # iterate over the rows and add them to the edge dict if the fraction is >= min_fraction
        for index, row in single_undirected_edge_df.iterrows():
            if row['fraction'] >= min_fraction:
                # create the edge string
                edge_str = f"{row['src']} {row['edge_type']} {row['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = row['fraction']
                pass

        # now lets process the undirected edges that have multiple rows
        multiple_undirected_edges_df = undirected_edges_df.groupby(['src', 'dest']).filter(lambda x: len(x) > 1)
        # iterate over two rows, and if the sum of fraction is greater than min_fraction then keep the edge
        for i in range(0, len(multiple_undirected_edges_df), 2):
            row_pairs = multiple_undirected_edges_df.iloc[i:i+2]
            # get the sum of the fraction
            fraction_sum = row_pairs['fraction'].sum()
            # check if the sum is greater than min_fraction
            if fraction_sum >= min_fraction:
                # create the edge string
                # we use the first row to create the edge string since it has the highest fraction
                edge_str = f"{row_pairs.iloc[0]['src']} {row_pairs.iloc[0]['edge_type']} {row_pairs.iloc[0]['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = float(fraction_sum)
                pass
            pass

        return list(selected_edges.keys())

    def read_prior_file(self, prior_file) -> list:
        """
        Read a prior file and return the contents as a list of strings
        Args:
            prior_file - string with the path to the prior file
            
        Returns:
            list - list of strings representing the contents of the prior file
        """
        if not os.path.exists(prior_file):
            raise FileNotFoundError(f"Prior file {prior_file} not found.")
        
        with open(prior_file, 'r') as f:
            self.prior_lines = f.readlines()
        
        return self.prior_lines

    def extract_knowledge(self, prior_lines) -> dict:
        """
        returns the knowledge from the prior file
        Args:
            prior_lines - list of strings representing the lines in the prior file
        Returns:
            dict - a dictionary where keys are
                addtemporal, forbiddirect, requiredirect
                 
                For addtemporal is a dictionary where the keys are the tier numbers (0 based) and 
                values are lists of the nodes in that tier.

                For forbiddirect and requiredirect, they will be empty in this case as this method is only for addtemporal.
        """
        tiers = {}
        inAddTemporal = False
        stop = False
        for line in prior_lines:
            # find the addtemporal line
            if line.startswith('addtemporal'):
                inAddTemporal = True
                continue
            # find the end of the addtemporal block
            if inAddTemporal and (line.startswith('\n') or line.startswith('forbiddirect')):
                inAddTemporal = False
                continue
            if inAddTemporal:
                # expect 1 binge_lag vomit_lag panasneg_lag panaspos_lag pomsah_lag

                # split the line
                line = line.strip()
                items = line.split()

                # add to dictionary
                if len(items) != 0:
                    tiers[int(items[0])-1] = items[1:]

        knowledge = {
            'addtemporal': tiers
        }

        return knowledge   

    def add_to_tier(self,tier,node):
        """
        Add to tier for prior knowledge, add temporal

        Args:
            tier (_type_): _description_
            node (_type_): _description_
        """
        
        self.knowledge.addToTier(self.lang.Integer(tier), self.lang.String(node))

    def clearKnowledge(self):
        """
        Clears the knowledge in the search object
        """
        self.knowledge = self.td.Knowledge()
        
    def load_knowledge(self, knowledge:dict):
        """
        Load the knowledge
        
        The standard prior.txt file looks like this:
        
        /knowledge

        addtemporal
        1 Q2_exer_intensity_ Q3_exer_min_ Q2_sleep_hours_ PANAS_PA_ PANAS_NA_ stressed_ Span3meanSec_ Span3meanAccuracy_ Span4meanSec_ Span4meanAccuracy_ Span5meanSec_ Span5meanAccuracy_ TrailsATotalSec_ TrailsAErrors_ TrailsBTotalSec_ TrailsBErrors_ COV_neuro_ COV_pain_ COV_cardio_ COV_psych_
        2 Q2_exer_intensity Q3_exer_min Q2_sleep_hours PANAS_PA PANAS_NA stressed Span3meanSec Span3meanAccuracy Span4meanSec Span4meanAccuracy Span5meanSec Span5meanAccuracy TrailsATotalSec TrailsAErrors TrailsBTotalSec TrailsBErrors COV_neuro COV_pain COV_cardio COV_psych

        forbiddirect

        requiredirect
        
        The input dict will have the keys of addtemporal, forbiddirect, requiredirect
        
        For the addtemporal key, the value will be another dict with the keys of 1, 2, 3, etc.
        representing the tiers. The values will be a list of the nodes in that tier.
        
        Args:
        search - search object
        knowledge - dictionary with the knowledge
        
        """
        
        # check if addtemporal is in the knowledge dict
        if 'addtemporal' in knowledge:
            tiers = knowledge['addtemporal']
            for tier, nodes in tiers.items():
                # tier is a number, tetrad uses 0 based indexing so subtract 1
                for node in nodes:
                    self.add_to_tier(tier, node)
                    pass

        # if there are other knowledge types, load them here
        pass

    def extract_edges(self, text) -> list:
        """
        Extract out the edges between Graph Edges and Graph Attributes
        from the output of the search.
        
        Args:
        text - text output from search
        
        Return:
        list of edges
        
        """
        edges = set()
        nodes = set()
        pairs = set()  # alphabetical order of nodes of an edge
        # get the lines
        lines = text.split('\n')
        startFlag=False  # True when we are in the edges, False when not
        for line in lines:
            # check if line begins with a number and period
            # convert line to python string
            line = str(line)
            if re.match(r"^\d+\.", line): # matches lines like "1. drinks --> happy"
                linestr = str(line)
                clean_edge = linestr.split('. ')[1]
                edges.add(clean_edge)
                
                # add nodes
                nodeA = clean_edge.split(' ')[0]
                nodes.add(nodeA)
                nodeB = clean_edge.split(' ')[2]
                nodes.add(nodeB)
                combined_string = ''.join(sorted([nodeA, nodeB]))
                pairs.add(combined_string)
                pass
        
        return list(edges)

    def summarize_estimates(self, df):
        """
        Summarize the estimates
        """
        # get the Estimate column from the df 
        estimates = df['Estimate']       
        # get the absolute value of the estimates
        abs_estimates = estimates.abs()
        # get the mean of the absolute values
        mean_abs_estimates = abs_estimates.mean()
        # get the standard deviation of the absolute values
        std_abs_estimates = abs_estimates.std()
        return {'mean_abs_estimates': mean_abs_estimates, 'std_abs_estimates': std_abs_estimates}
        
    def edges_to_lavaan(self, edges, exclude_edges = ['---','<->','o-o']):
        """
        Convert edges to a lavaan string
        """
        lavaan_model = ""
        for edge in edges:
            nodeA = edge.split(' ')[0]
            nodeB = edge.split(' ')[2]
            edge_type = edge.split(' ')[1]
            if edge_type in exclude_edges:
                continue
            # remember that for lavaan, target ~ source
            lavaan_model += f"{nodeB} ~ {nodeA}\n"
        return lavaan_model
    
    def run_semopy(self, lavaan_model, data):  
        
        """
        run sem using semopy package
        
        lavaan_model - string with lavaan model
        data - pandas df with data
        """
        
        # create a sem model   
        model = semopy.Model(lavaan_model)

        ## TODO - check if there is a usable model,
        ## for proj_dyscross2/config_v2.yaml - no direct edges!
        ## TODO - also delete output files before writing to them so that
        ## we don't have hold overs from prior runs.
        opt_res = model.fit(data)
        estimates = model.inspect()
        stats = semopy.calc_stats(model)
        
        # change column names lval to dest and rval to src
        estimatesRenamed = estimates.rename(columns={'lval': 'dest', 'rval': 'src'})
        # convert the estimates to a dict using records
        estimatesDict = estimatesRenamed.to_dict(orient='records')        

        return ({'opt_res': opt_res,
                 'estimates': estimates, 
                 'estimatesDict': estimatesDict,
                 'stats': stats,
                 'model': model})

    def add_sem_results_to_graph(self, obj, df, format: bool = True):
        """
        Add the semopy results to the graph object.
        Args:
            obj (DgraphFlex): The graph object to add the results to.
            df (pd.DataFrame): The semopy results dataframe.
            format (bool): Whether to format the estimates or not. Defaults to True.
        
        """
        # iterate over the estimates id the df semopy output
        # and add them to the graph object

        # iterate over the estimates and add them to the graph
        # object
        
        for index, row in df.iterrows():
            if row['op'] == '~':
                source = row['rval']
                target = row['lval']
                estimate = row['Estimate']
                pvalue = row['p-value']
                # modify the edge in the existing graph obj
                # set the color based on the sign of estimate, green for positive, red for negative
                color = 'green' if estimate > 0 else 'red'
                
                obj.modify_existing_edge(source, target, color=color, strength=estimate, pvalue=pvalue)
                pass

    def run_model_search(self, df, **kwargs) -> tuple:
        """
        Run a search
        
        Args:
        df - pandas dataframe
        
        kwargs:
        model - string with the model to use, default gfci
        knowledge - dictionary with the knowledge
        score - dictionary with the arguments for the score
            e.g. {"sem_bic": {"penalty_discount": 1}}
            
        test - dictionary with the arguments for the test alpha 
        graph - boolean, if False, will not return a graph object
        run_sem - boolean, if True, will run semopy on the results
        jitter - boolean, if True, will jitter the data
        
        Returns:
        result - dictionary with the results
        dg - DgraphFlex object with the edges and sem results
        """
    
        model = kwargs.get('model', 'gfci')
        knowledge = kwargs.get('knowledge', None)
        score = kwargs.get('score', None)
        test = kwargs.get('test', None)
        run_graph = kwargs.get('run_graph', True)
        run_sem = kwargs.get('run_sem', True)
        jitter = kwargs.get('jitter',False)
        depth = kwargs.get('depth', -1)
        verbose = kwargs.get('verbose', False)
        max_degree = kwargs.get('max_degree', -1)
        max_disc_path_length = kwargs.get('max_disc_path_length', -1)
        complete_rule_set_used = kwargs.get('complete_rule_set_used', False)
        guarantee_pag = kwargs.get('guarantee_pag', False)
        
        # check if score is not None
        if score is not None:  
            ## Use a SEM BIC score
            if 'sem_bic' in score:
                penalty_discount = score['sem_bic']['penalty_discount']
                
        if test is not None:
            if 'fisher_z' in test:
                alpha = test['fisher_z'].get('alpha',.01)
            
        # check if depth is not None - set in run_gfci
        # if depth != -1:
        #     self.set_depth(depth)
            
        if knowledge is not None:
            self.load_knowledge(knowledge)
        
        # set the verbosity
        # if verbose == False:
        #     self.params.set(Params.VERBOSE, False)

        ## Run the selected search
        if model == 'gfci':   
                
            graph = self.run_gfci(  df,
                                alpha=alpha,
                                penalty_discount=penalty_discount,
                                jitter=jitter
                            )

        edges = self.extract_edges(graph)
        
        if run_graph:
            # create the graph object
            dg = DgraphFlex()
            # add the edges to the graph object
            dg.add_edges(edges)
            # if run_sem is True, then run semopy on the edges
            if run_sem:
                # convert edges to lavaan model
                lavaan_model = self.edges_to_lavaan(edges)
                # run semopy on the lavaan model
                sem_results = self.run_semopy(lavaan_model, df)
                # add the sem results to the graph object
                self.add_sem_results_to_graph(dg, sem_results['estimates'])

        else:
            sem_results = None
            dg = None
            
        result = {  'edges': edges,
                    'cda_output': graph,
                    'sem_results': sem_results,
                  } 
        
        return result, dg

    def run_stability_search(self, full_df: pd.DataFrame,
                            model: str = 'gfci',
                            knowledge: Optional[dict] = None,
                            score = {'sem_bic': {'penalty_discount': 1.0}},
                            test ={'fisher_z': {'alpha': .05}},
                            runs: int = 100,
                            min_fraction: float= 0.75,
                            subsample_fraction: float = 0.9,
                            random_state: Optional[int] = None,
                            lag_stub: str = '',
                            save_file: Optional[str] = None) -> tuple:
        """
        Run a stability search on the DataFrame using the specified model.
        
        Edges are tabluated for each run using a set.
        Edges that are present for a minimum of min_fraction of runs will be retained
        and returned.
        The edges are returned as a list of strings.
        
        Args:
            df: pd.DataFrame - the DataFrame to perform the stability search on
            model: str - the model to use for the search
            knowledge: Optional[dict] - additional knowledge to inform the search
            score: dict - scoring parameters
            test: dict - testing parameters
            runs: int - number of runs to perform
            min_fraction: float - minimum fraction of runs an edge must appear in to be retained
            subsample_fraction: float - fraction of data to subsample for each run
            random_state: Optional[int] - random state for reproducibility
            lag_stub: - if given a string, will add lagged columns to the DataFrame with the stub
        Returns:
            list - list with the results
        """

        # dictionary where key is the edge and value is the number of times it was found
        edge_counts = {}
        
        # list to hold results of each run
        run_results = []

        # check if in jupyter to select the progress bar code
        if self.in_jupyter():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm      
              
        # loop over the number of runs
        myRuns = range(runs)
        for i in tqdm(myRuns, desc=f"Running stability search with {runs} runs", unit="run"):
            # subsample the DataFrame
            df = self.subsample_df(full_df, fraction=subsample_fraction, random_state=random_state)
            
            # check if lag_flag is True
            if lag_stub:
                # add lagged columns
                df = self.add_lag_columns(df, lag_stub=lag_stub)
                
            # standardize the data
            df = self.standardize_df_cols(df)
                
            # run the search, no sem
            searchResult,_ = self.run_model_search(df, model=model, 
                                                knowledge=knowledge, 
                                                score=score,
                                                test=test,
                                                run_graph=False,
                                                verbose=True)
            # get the edges
            edges = searchResult['edges']
            # loop over the edges
            for edge in edges:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

            info = {
                'edges': edges,
                'edge_counts': edge_counts,
            }
            # add the info to the runs list
            run_results.append(info)
        print(f"\nSearch complete!")
        
        # check similarity of edges, sort alphabetically
        # get all the keys and then sort them

        sorted_edge_keys = sorted(edge_counts.keys())
        
        sorted_edge_counts = {}
        sorted_edge_counts_raw = {}
        # loop over the sorted keys and store the fractional count 
        for edge in sorted_edge_keys:
            sorted_edge_counts[edge] = edge_counts[edge]/runs
            sorted_edge_counts_raw[edge] = edge_counts[edge]

        selected_edges = self.select_edges(sorted_edge_counts, min_fraction=min_fraction)

        # combine results into a dictionary
        results = {
            'selected_edges': selected_edges,
            'sorted_edge_counts': sorted_edge_counts,
            'sorted_edge_counts_raw': sorted_edge_counts_raw,
            'edge_counts': edge_counts, 
            #'run_results': run_results, # error is not JSON serializable
        }

        if save_file is not None:
            # save the results to a json file
            with open(save_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {save_file}")

        return selected_edges, sorted_edge_counts, sorted_edge_counts_raw, run_results
    def in_jupyter(self)->bool:
        """
        Check if the code is running in a Jupyter Notebook environment.

        Returns:
            bool: True if running in a Jupyter Notebook, False otherwise.
        
        """
        try:
            # Check if the IPython module is available
            from IPython import get_ipython
            ipython = get_ipython()
            
            # Check if the current environment is a Jupyter Notebook
            if ipython is not None and 'IPKernelApp' in ipython.config:
                return True
        except ImportError:
            pass
        
        return False
            
    def run_gfci(self, df: pd.DataFrame,
                 alpha: float = 0.01,
                 penalty_discount: float = 1,
                 jitter: bool = False) -> str:
        
        data = self.df_to_data(df, jitter)

        test = self.ts.test.IndTestFisherZ(data, alpha)
        score = self.ts.score.SemBicScore(data, True)
        score.setPenaltyDiscount(penalty_discount)
        score.setStructurePrior(0)

        # FOR THE MOST PART, DONT CHANGE ANY OF THESE
        # UNLESS COMPUTATION IS TAKING TOO LONG
        gfci = self.ts.GFci(test, score)
        gfci.setCompleteRuleSetUsed(True)
        gfci.setDepth(-1)
        gfci.setDoDiscriminatingPathRule(True)
        gfci.setFaithfulnessAssumed(True)
        gfci.setMaxDegree(-1)
        gfci.setMaxPathLength(-1)
        gfci.setPossibleMsepSearchDone(True)
        gfci.setVerbose(False)
        
        # set knowledge
        gfci.setKnowledge(self.knowledge)
        
        # run the search
        graph = gfci.search().toString()    
        
        return graph

    def test1(self):
        """
        Test running a simple gfci model
        """
        df = self.read_csv('data/boston_data_raw.csv')
        output=self.run_gfci(df)
        edges = self.extract_edges(output)
        pass
            
    def test(self):
        """
        Test running a simple gfci model
        with lag
        """
        df = self.read_csv('data/boston_data_raw.csv')
        # add lag
        df = self.add_lag_columns(df, lag_stub='_lag')
        # standardize the columns
        df = self.standardize_df_cols(df)
        
        # load prior
        prior_lines = self.read_prior_file(f'data/boston_prior.txt')
        # extract knowledge
        knowledge = self.extract_knowledge(prior_lines)

        self.load_knowledge(knowledge)
        
        output=self.run_gfci(df)
        edges = self.extract_edges(output)

        # get lavaan_model
        lavaan_model = self.edges_to_lavaan(edges)
        # run the semopy
        sem_results = self.run_semopy(lavaan_model, df)
        
        # create the graph
        obj = DgraphFlex()
        obj.add_edges(edges)
        # output graph
        obj.save_graph("png","test_plot")
                
        # add sem info to graph
        self.add_sem_results_to_graph(obj,sem_results['estimates'])
        # output graph
        obj.save_graph("png","test_plot_sem")    
        
        # create the graph excluding non directed edges
        obj = DgraphFlex()
        obj.add_edges(edges, exclude = ['---','o-o','<->'])
        # output graph
        obj.save_graph("png","test_plot2")
                
        # add sem info to graph
        self.add_sem_results_to_graph(obj,sem_results['estimates'])
        # output graph
        obj.save_graph("png","test_plot2_sem")  
            
        pass
        
if __name__ == "__main__":
    
    tp = TetradPlus()
    tp.test()
    pass