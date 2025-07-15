import taosws
import pandas as pd
from datetime import timedelta
import os

# change to your own
datapath = "train_data_upload"
output_folder = 'train_data_download'
database = "train_Data"
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
        print(f"Database {database} set up successfully.")
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
        for column in columns:
            dtype = df[column].dtype
            if pd.api.types.is_float_dtype(dtype):
                schema_columns.append(f"`{column.replace(' ', '_')}` FLOAT")
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

print(setup_tables.__code__.co_consts)



# insert_data_from_csv() function to insert data from CSV files into tables.
def insert_data_from_csv(conn, database, csv_file, table_name):
    try:
        cursor = conn.cursor()
        df = pd.read_csv(csv_file)

        setup_tables(conn, database, table_name, df)
        cursor.execute(f"USE {database}")

        current_time = pd.Timestamp.now()  # Start with the current timestamp

        for _, row in df.iterrows():
            # Format the current timestamp and ensure uniqueness by incrementing it for each row
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

            insert_query = f"INSERT INTO {table_name} VALUES({', '.join(values)})"
            print(f"Inserting data: {insert_query}")
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

        # Get column names and remove 'ts'
        cursor.execute(f"DESCRIBE {table_name}")
        columns = [desc[0] for desc in cursor.fetchall()]

        # Remove 'ts' from columns list if it exists
        if 'ts' in columns:
            ts_index = columns.index('ts')
            columns.remove('ts')

            # Remove the 'ts' column from each row in data
            data = [tuple(item for i, item in enumerate(row) if i != ts_index) for row in data]

        # Create the DataFrame with columns excluding 'ts'
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
