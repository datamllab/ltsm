import unittest
from unittest.mock import patch, MagicMock
import ltsm.data_reader.database_reader as db_connector
import pandas as pd

class TestDatabaseConnector(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame with different data types
        self.input_df = pd.DataFrame({
            'Updated Time': ['06/30/2023 19:01:24', '06/30/2023 19:03:04', '06/30/2023 19:04:44'],
            'Temperature': [61.71, 69.21,323.64],  # Float
            'Count': [10, 15,18],  # Integer
            'Status': [True, False,False],  # Boolean
            'Description': ['Normal', 'High','Low']  # String
        })
        self.database = "test_database"
        self.table_name = "test_table"

    @patch('ltsm.data_reader.database_reader.create_connection')
    def test_setup_tables_with_various_data_types(self, mock_create_connection):
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_create_connection.return_value = mock_conn

        # Call the function to be tested
        db_connector.setup_tables(mock_conn, self.database, self.table_name, self.input_df)

        # Check if the correct SQL commands were executed
        mock_cursor.execute.assert_any_call(f"USE {self.database}")
        expected_schema = "(ts TIMESTAMP, Temperature FLOAT, Count INT, Status BOOL, Description STRING)"
        mock_cursor.execute.assert_any_call(f"CREATE TABLE IF NOT EXISTS {self.table_name} {expected_schema}")

    @patch('ltsm.data_reader.database_reader.create_connection')
    @patch('ltsm.data_reader.database_reader.pd.read_csv')
    def test_insert_data_with_various_data_types(self, mock_read_csv, mock_create_connection):
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_create_connection.return_value = mock_conn

        # Mock reading CSV with various data types
        mock_read_csv.return_value = self.input_df

        # Call the function to be tested
        db_connector.insert_data_from_csv(mock_conn, self.database, "dummy_path.csv", self.table_name)

        # Check if data insertion commands were executed
        self.assertTrue(mock_cursor.execute.called)
        self.assertEqual(mock_cursor.execute.call_count, len(self.input_df)+4)  # Check the number of execute calls

if __name__ == '__main__':
    unittest.main()
