import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import taosws
import os

# Assuming your script is named `script` and it contains the functions defined
from ltsm.data_reader.npy_database_reader import create_connection, setup_database, setup_tables, insert_data_from_npy, retrieve_data_to_npy

class TestDatabaseOperations(unittest.TestCase):

    def setUp(self):
        # Simulated database connection
        self.conn = MagicMock(spec=taosws.Connection)
        self.cursor = MagicMock()
        self.conn.cursor.return_value = self.cursor

        # Generate a large, complex synthetic NumPy array for testing (1000 rows, 50 columns)
        self.num_rows = 1000
        self.num_cols = 50
        np.random.seed(42)
        self.data = np.random.rand(self.num_rows, self.num_cols) # Random floats between 0 and 100

        # Save the array to a temporary .npy file
        self.test_npy_file = 'test_data.npy'
        np.save(self.test_npy_file, self.data)

        # Create a DataFrame from the NumPy array for table setup
        self.df = pd.DataFrame(self.data)
        self.table_name = 'test_table'

    @patch('taosws.connect')
    def test_create_connection(self, mock_connect):
        # Test the connection creation
        mock_connect.return_value = self.conn
        connection = create_connection()
        mock_connect.assert_called_once()
        self.assertIsNotNone(connection)

    def test_setup_database(self):
        # Test database setup
        setup_database(self.conn, 'test_database')
        self.cursor.execute.assert_called_with("CREATE DATABASE IF NOT EXISTS test_database")

    def test_setup_tables(self):
        # Test table creation with a large number of columns
        setup_tables(self.conn, 'test_database', self.table_name, self.df)
        self.cursor.execute.assert_any_call(f"USE test_database")
        self.cursor.execute.assert_any_call(f"DROP TABLE IF EXISTS {self.table_name}")

        # Check that the table creation schema was executed correctly
        expected_schema_columns = [f"`{i}` FLOAT" for i in range(self.num_cols)]
        expected_schema = f"CREATE TABLE IF NOT EXISTS {self.table_name} (ts TIMESTAMP, {', '.join(expected_schema_columns)})"
        self.cursor.execute.assert_any_call(expected_schema)

    def test_insert_data_from_npy(self):
        # Test data insertion from .npy file with batch processing
        insert_data_from_npy(self.conn, 'test_database', self.test_npy_file, self.table_name, batch_size=100)
        self.cursor.execute.assert_any_call(f"USE test_database")

        # Check that data is inserted in batches
        batch_inserts = [call for call in self.cursor.execute.call_args_list if "INSERT INTO" in call[0][0]]
        self.assertGreaterEqual(len(batch_inserts), 5, "Expected at least 5 batch insertions for 1000 rows with a batch size of 200.")

    def test_retrieve_data_to_npy(self):
        # Mock fetched data and column descriptions for the retrieval test
        self.cursor.fetchall.side_effect = [
            [tuple(row) for row in self.data],  # Mocked data returned as tuples
            [(f'{i}',) for i in range(self.num_cols)]  # Mocked column names
        ]

        output_file = 'test_output.npy'
        retrieve_data_to_npy(self.conn, 'test_database', self.table_name, output_file)

        # Verify that the SELECT command was called
        self.cursor.execute.assert_any_call(f"SELECT * FROM {self.table_name}")

        # Load and check the output file
        result_array = np.load(output_file)
        self.assertEqual(result_array.shape, (self.num_rows, self.num_cols),    # Subtract 1 for the timestamp column
                         "Output file shape does not match expected data shape.")
        np.testing.assert_array_almost_equal(self.data, result_array, decimal=5,
                                             err_msg="Output data does not match expected data.")

        # Clean up by deleting the generated .npy file
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Cleaned up generated file: {output_file}")

        if os.path.exists('test_data.npy'):
            os.remove('test_data.npy')
            print(f"Cleaned up generated file: test_data.npy")

if __name__ == '__main__':
    unittest.main()
