import unittest
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import MagicMock, patch
import taosws

# Assuming your script is named `script` and contains the functions
from ltsm.data_reader.train_database_reader import create_connection, setup_database, setup_tables, insert_data_from_csv, retrieve_data_to_csv

class TestDatabaseOperations(unittest.TestCase):

    def setUp(self):
        # Simulated database connection
        self.conn = MagicMock(spec=taosws.Connection)
        self.cursor = MagicMock()
        self.conn.cursor.return_value = self.cursor

        #  A larger, complex CSV input data (1000 rows, 10 float columns, 1 int column as Label)
        num_rows = 1000
        num_features = 10
        np.random.seed(42)
        float_data = np.random.rand(num_rows, num_features)  # Generate random float values between 0 and 1
        label_data = np.random.randint(0, 2, size=(num_rows, 1))  # Random integer values 0 or 1 for 'Label'

        # Combine float data and label column to create a full dataset
        data = np.hstack((float_data, label_data))
        columns = [f'Feature{i + 1}' for i in range(num_features)] + ['Label']

        # Create a Pandas DataFrame from the generated data
        self.df = pd.DataFrame(data, columns=columns)

        # Ensure 'Label' is an integer type
        self.df['Label'] = self.df['Label'].astype(int)

        self.input_csv = StringIO(self.df.to_csv(index=False))

        # Sample expected table creation schema
        self.expected_schema = (
                "CREATE TABLE IF NOT EXISTS test_table ("
                "ts TIMESTAMP, " +
                ", ".join([f"`Feature{i + 1}` FLOAT" for i in range(num_features)]) +
                ", Label INT)"
        )

    @patch('taosws.connect')
    def test_create_connection(self, mock_connect):
        # Test connection creation
        mock_connect.return_value = self.conn
        connection = create_connection()
        mock_connect.assert_called_once()
        self.assertIsNotNone(connection)

    def test_setup_database(self):
        # Test database setup
        setup_database(self.conn, 'test_database')
        self.cursor.execute.assert_called_with("CREATE DATABASE IF NOT EXISTS test_database")

    def test_setup_tables(self):
        # Test table creation
        setup_tables(self.conn, 'test_database', 'test_table', self.df)
        self.cursor.execute.assert_any_call(f"USE test_database")
        self.cursor.execute.assert_any_call(self.expected_schema)

    def test_insert_data_from_csv(self):
        # Test data insertion
        insert_data_from_csv(self.conn, 'test_database', self.input_csv, 'test_table')
        self.cursor.execute.assert_any_call(f"USE test_database")
        # Check that data is being inserted
        self.assertTrue(
            any("INSERT INTO test_table VALUES" in call[0][0] for call in self.cursor.execute.call_args_list),
            "Insert data command was not called correctly."
        )

    def test_retrieve_data_to_csv(self):
        # Mock fetched data and column descriptions
        self.cursor.fetchall.side_effect = [
            [tuple(row) for row in self.df.values],  # Fetched data as tuples
             [(f'Feature{i+1}',) for i in range(10)] + [('Label',)]  # Column names
        ]

        output_file = "test_output.csv"
        retrieve_data_to_csv(self.conn, 'test_database', 'test_table', output_file)

        # Verify that the SELECT command was called
        self.cursor.execute.assert_any_call(f"SELECT * FROM test_table")

        # Check if output file is created and matches expected data structure
        result_df = pd.read_csv(output_file)
        self.assertEqual(len(result_df), 1000, "Output file does not have the expected number of rows.")
        for i in range(1, 11):
            self.assertTrue(f'Feature{i}' in result_df.columns, f"Expected column Feature{i} not found in output CSV.")
        self.assertTrue('Label' in result_df.columns, "Expected column 'Label' not found in output CSV.")

if __name__ == '__main__':
    unittest.main()