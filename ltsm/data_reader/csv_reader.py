import os
import pandas as pd
import logging
from pathlib import Path
from ltsm.common.base_reader import BaseReader
from typing import Any

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
)

def transform_csv(input_file: str) -> pd.DataFrame:
    """
    This function reads the CSV file, deletes the first row and the first column, 
    replaces the time with 0, 1, 2 sequence, and returns the transformed DataFrame.
    
    Args:
        input_file(str): CSV file path

    Returns:
        pd.DataFrame: transformed DataFrame
    """
    try:
        df = pd.read_csv(input_file, header=None)
        df = df.drop(index=0)
        if df.shape[1] > 1:  
            df = df.drop(columns=[df.columns[0]])
        df = df.fillna(0)  # deal with possible NaN values
        df_transposed = df.T 
        df_transposed.columns = range(len(df_transposed.columns))

        return df_transposed.reset_index(drop=True) # reset index to start from 0
    except FileNotFoundError as e:
        logging.error(f"Error: File not found - {input_file}")
        raise e  

    except pd.errors.EmptyDataError as e:
        logging.error(f"Error: The file is empty or invalid - {input_file}")
        raise e

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e

def transform_csv_dataset(input_folder: str, output_folder: str):
    """
    Iterates through all the CSV files in the input folder and 
    converts each one, saving it to the output folder.

    Args:
        input_folder(str): path to the folder containing the CSV files to be transformed
        output_folder(str): output folder path

    Returns:
        list of transformed DataFrames
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    rtn_data = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            try:
                df_transformed = transform_csv(input_file)
                df_transformed.to_csv(output_file, index=False)
                rtn_data.append(df_transformed)
                logging.info(f"DK CSV transform finished. Output saved to {output_file}")
            except Exception as e:
                print("here")
                logging.error(f"Processing {input_file} , have error: {e}")
                raise e
    return rtn_data


class CSVReader(BaseReader):
    """
    Represents a CSV Reader object for processing time-series data. 
    
    This class reads .csv files, fills missing values using linear interpolation, and drops any invalid columns. 
    It assumes that the .csv file's columns represent time-series data, while each row corresponds to a data
    instance or feature. 

    Attributes:
        module_id (str): The identifier for the base reader objects.
        data_path (str): The file path where the .csv file is located.
    """
    module_id = "csv"
    def __init__(self, data_path: str):
        """
        Initializes the CSVReader class.

        Args:
            data_path (str): The file path where the .csv file is located.
        """
        super().__init__()
        self.data_path = data_path

    def fetch(self) -> pd.DataFrame:
        """
        Fetches data in the .csv file specified at data_path.

        Reads the .csv file, fills missing values, and drops invalid columns.

        Returns:
            pd.DataFrame: The data from the .csv as a DataFrame.
        """

        # Check if file exists
        if not Path(self.data_path).is_file():
            raise FileNotFoundError(f"File not found at the specified path: {self.data_path}")

        # Read data, extract columns, toss non-datetime columns
        try:
            loaded_data = pd.read_csv(self.data_path, header=None)
            
            # loaded_data = loaded_data.drop(index=0)
            loaded_data.columns = loaded_data.iloc[0]
            loaded_data = loaded_data[1:]
            loaded_data.reset_index(drop=True, inplace=True)
            # Transpose the data if each time-series sequence is saved in the columns
            if loaded_data.shape[1] < loaded_data.shape[0]:
                # Assuming if there are more rows than columns, then time-series sequence is saved in columns
                if loaded_data.shape[1] > 1: 
                    # Drop first column containing time-series indices
                    loaded_data = loaded_data.drop(columns=[loaded_data.columns[0]])
                loaded_data = loaded_data.T 
                #loaded_data.columns = range(len(loaded_data.columns))
            loaded_data.index.name = None
            loaded_data.columns.name = None
            loaded_data.columns = range(len(loaded_data.columns))
            #loaded_data = loaded_data.reset_index(drop=True) # reset index to start from 0
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file at {self.data_path} is empty.")
        except pd.errors.ParserError:
            raise ValueError(f"Failed to parse CSV file at {self.data_path}.")
        except Exception as e:
            raise e

        for col in loaded_data.columns:
            if not pd.api.types.is_float_dtype(loaded_data[col]):
                # Try to convert to numeric data type
                try:
                    loaded_data[col]= pd.to_numeric(loaded_data[col])
                except Exception as e:
                    # Drop columns that do not contain float data
                    logging.info(f"Dropping column '{col}' as it does not contain float data.")
                    loaded_data.drop(columns=col, inplace=True)

        # Function to convert float-like strings to integer strings where possible
        def convert_float_index(index):
            try:
                # Try to convert the index to a float and then to an integer if it's a whole number
                float_val = float(index)
                int_val = int(float_val)
                # Check if the float and integer representations are equivalent
                return int_val if float_val == int_val else index
            except ValueError:
                # If conversion fails, return the original index
                return index

        # Apply the function to each index
        loaded_data.index = loaded_data.index.map(convert_float_index)

        # Fill NA through linear interpolation
        def fillna(row):
            if row.isna().any():
                return row.interpolate(method='linear', limit_direction='both', inplace=False)
            return row

        loaded_data = loaded_data.apply(fillna, axis=1)
        return loaded_data
        
if __name__ == '__main__':
    input_folder = './datasets/DK/'
    output_folder = './datasets/DK_transformed/'
    transform_csv_dataset(input_folder, output_folder)