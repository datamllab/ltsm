import taosws
import pandas as pd
import os
import numpy as np
from datetime import timedelta
# change to your own
datapath = "time_series_preds_upload"
output_folder = 'time_series_preds_download'
database = "time_series_preds"
user = "root"
password = "taosdata"
insert_batch_size = 100

# create_connection() function to connect to the database. (change host and port to your own)
def create_connection(host='35.153.211.255', port=6041):
    conn = None
    try:
        conn = taosws.connect(
            user=user,
            password=password,
            host=host,
            port=port,
        )
        print(f"Connected to {host}:{port} successfully.")
        return conn
    except Exception as err:
        print(f"Failed to connect to {host}:{port}, ErrMessage: {err}")
        raise err


# setup_database() function to create a new database if it doesn't exist.
def setup_database(conn, database):
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        print(f"Database {database} set up successfully.")
    except Exception as err:
        print(f"Error setting up database: {err}")
        raise err


# setup_tables() function to create tables based on NPY column names and data types.
def setup_tables(conn, database, table_name, df):
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {database}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        columns = df.columns
        schema_columns = ["ts TIMESTAMP"]
        print(df)
        # Infer column types and set schema accordingly
        schema_extend_length=df.shape[1]
        for i in range(schema_extend_length):
            schema_columns.append(f"`{i}` FLOAT")
        schema = f"({', '.join(schema_columns)})"

        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {schema}")
        print(f"Table {table_name} set up successfully with schema: {schema}")
    except Exception as err:
        print(f"Error setting up database or table {table_name}: {err}")
        raise err

print(setup_tables.__code__.co_consts)



# insert_data_from_npy() function to insert data from NPY files into tables.
def insert_data_from_npy(conn, database, npy_file, table_name, batch_size=insert_batch_size):
    try:
        cursor = conn.cursor()
        data = np.load(npy_file)
        df = pd.DataFrame(data)
        print(f"input npy {table_name}:{df}")
        # Create table if needed
        setup_tables(conn, database, table_name, df)
        cursor.execute(f"USE {database}")

        current_time = pd.Timestamp.now()  # Start with the current timestamp
        rows = []

        for i, row in df.iterrows():
            values = [f"'{current_time.strftime('%Y-%m-%d %H:%M:%S')}'"]  # Current timestamp value
            current_time += timedelta(seconds=1)  # Increment timestamp by 1 second

            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    values.append("NULL")
                elif isinstance(value, str):
                    values.append(f"'{value}'")
                elif isinstance(value, bool):
                    values.append("true" if value else "false")
                else:
                    values.append(str(value))

            # Append the formatted row to the batch
            rows.append(f"({', '.join(values)})")

            # If batch size is reached, execute the batch insert
            if len(rows) >= batch_size:
                insert_query = f"INSERT INTO {table_name} VALUES " + ", ".join(rows)
                print(f"Inserting batch of {batch_size} rows")
                cursor.execute(insert_query)
                rows = []  # Reset the batch

        # Insert remaining rows if any
        if rows:
            insert_query = f"INSERT INTO {table_name} VALUES " + ", ".join(rows)
            print(f"Inserting final batch of {len(rows)} rows")
            cursor.execute(insert_query)

        print(f"Data from {npy_file} inserted into table {table_name} successfully.")
    except Exception as err:
        print(f"Error inserting data from {npy_file} into {table_name}: {err}")
        raise err



# retrieve_data_to_npy() function to retrieve data from a table and save it to a NPY file.
def retrieve_data_to_npy(conn, database, table_name, output_file):
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {database}")
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        cursor.execute(f"DESCRIBE {table_name}")

        df = pd.DataFrame(data)
        if 0 in df.columns and pd.api.types.is_datetime64_any_dtype(df[0]):
            df.drop(columns=[0], inplace=True)
        data_array = df.to_numpy()
        np.save(output_file, data_array)
        print(f"Data from {table_name} saved to {output_file}.")
        print(f"{table_name}:{df}")
    except Exception as err:
        print(f"Error retrieving data from {table_name}: {err}")
        raise err


# Example usage
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        try:
            setup_database(conn, database)
            npy_files = [os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith('.npy')]
            tables = [os.path.splitext(os.path.basename(npy_file))[0] for npy_file in npy_files]
            for npy_file, table_name in zip(npy_files, tables):
                insert_data_from_npy(conn, database, npy_file, table_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for table_name in tables:
                output_file = os.path.join(output_folder, f"{table_name}.npy")
                retrieve_data_to_npy(conn, database, table_name, output_file)

        finally:
            conn.close()
