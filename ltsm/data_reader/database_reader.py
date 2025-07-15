import taosws
import pandas as pd
import os

# change to your own
datapath = "original_upload"
output_folder = 'original_download'
database = "time_series_demo"
user = "root"
password = "taosdata"

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
        print("Database time_series_demo set up successfully.")
    except Exception as err:
        print(f"Error setting up database: {err}")
        raise err


# setup_tables() function to create tables based on CSV column names and data types.
def setup_tables(conn, database, table_name, df):
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {database}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        columns = df.columns
        schema_columns = ["ts TIMESTAMP"]

        # Infer column types and set schema accordingly
        for column in columns[1:]:
            dtype = df[column].dtype
            if pd.api.types.is_float_dtype(dtype):
                schema_columns.append(f"{column.replace(' ', '_')} FLOAT")
            elif pd.api.types.is_integer_dtype(dtype):
                schema_columns.append(f"{column.replace(' ', '_')} INT")
            elif pd.api.types.is_bool_dtype(dtype):
                schema_columns.append(f"{column.replace(' ', '_')} BOOL")
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema_columns.append(f"{column.replace(' ', '_')} TIMESTAMP")
            else:  # Treat as STRING for other types like object (text)
                schema_columns.append(f"{column.replace(' ', '_')} STRING")

        schema = f"({', '.join(schema_columns)})"
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {schema}")
        print(f"Table {table_name} set up successfully with schema: {schema}")
    except Exception as err:
        print(f"Error setting up database or table {table_name}: {err}")
        raise err


# insert_data_from_csv() function to insert data from CSV files into tables.
def insert_data_from_csv(conn, database, csv_file, table_name):
    try:
        cursor = conn.cursor()
        df = pd.read_csv(csv_file)

        # Ensure the first column is a timestamp
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')

        setup_tables(conn, database, table_name, df)
        cursor.execute(f"USE {database}")

        for _, row in df.iterrows():
            values = [f"'{row[df.columns[0]]}'"]  # Timestamp value
            for col in df.columns[1:]:
                value = row[col]
                if pd.isna(value):
                    values.append("NULL")
                elif isinstance(value, str):
                    values.append(f"'{value}'")
                elif isinstance(value, bool):
                    values.append("true" if value else "false")
                else:
                    values.append(str(value))

            insert_query = f"INSERT INTO {table_name} VALUES({', '.join(values)})"
            print(insert_query)
            cursor.execute(insert_query)

        print(f"Data from {csv_file} inserted into table {table_name} successfully.")
    except Exception as err:
        print(f"Error inserting data from {csv_file} into {table_name}: {err}")
        raise err


# retrieve_data_to_csv() function to retrieve data from a table and save it to a CSV file.
def retrieve_data_to_csv(conn, database, table_name, output_file):
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {database}")
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        cursor.execute(f"DESCRIBE {table_name}")
        columns = [desc[0] for desc in cursor.fetchall()]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Data from {table_name} saved to {output_file}.")
    except Exception as err:
        print(f"Error retrieving data from {table_name}: {err}")
        raise err


# Example usage
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        try:
            setup_database(conn, database)
            csv_files = [os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith('.csv')]
            tables = [os.path.splitext(os.path.basename(csv_file))[0] for csv_file in csv_files]
            for csv_file, table_name in zip(csv_files, tables):
                insert_data_from_csv(conn, database, csv_file, table_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for table_name in tables:
                output_file = os.path.join(output_folder, f"{table_name}.csv")
                retrieve_data_to_csv(conn, database, table_name, output_file)

        finally:
            conn.close()
