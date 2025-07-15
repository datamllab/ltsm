from ltsm.data_reader.csv_reader import CSVReader,transform_csv, transform_csv_dataset
import pytest
from pandas.api.types import is_float_dtype
import os
import pandas as pd
import numpy as np

@pytest.fixture
def setup_csvreader_data():
    dfs = [
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In'],
            ['6/30/2023 19:01:24', 0, 1, 98.34],
            ['6/30/2023 19:03:04', 0, 3, 98.93],
            ['6/30/2023 19:04:44', 0, 2, 97.90],
            ['6/30/2023 19:06:22', 2, 3, 98.37],
            ['6/30/2023 19:08:03', 3, 1, 98.37]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
            ['6/30/2023 19:09:43', 1, 2, 98.43, 109.31],
            ['6/30/2023 19:11:23', 1, 3, 97.64, 109.18],
            ['6/30/2023 19:13:02', 1, 4, np.nan, 109.18],
            ['6/30/2023 19:15:32', 1, 5, np.nan, np.nan],
            ['6/30/2023 19:17:41', 1, 6, 95.32, 109.18],
            ['6/30/2023 19:17:41', 1, np.nan, 95.32, 110.20]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp'],
            ['6/30/2023 19:01:24', 61.71, 102.75],
            ['6/30/2023 19:03:04', 69.21, 103.19],
            ['6/30/2023 19:04:44', 71.35, 103.34],
            ['6/30/2023 19:06:22', 68.14, 102.60],
            ['6/30/2023 19:08:03', 83.47, 103.26],
            ['6/30/2023 19:09:43', 63.14, 103.85]
        ]),
        pd.DataFrame([
            [0, 1, 2, 3, 4, 5, 6],
            [61.71, 102.75, np.nan, 109.73, 110.81, 111.32, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 98.30],
            [55.31, 71.35, 103.34, np.nan, 108.87, 109.32],
            [np.nan, np.nan, 100.32, 102.60, 98.37, np.nan, np.nan],
        ])
    ]

    dfs_expected = [
        pd.DataFrame({
            0: [0, 1, 98.34],
            1: [0, 3, 98.93],
            2: [0, 2, 97.90],
            3: [2, 3, 98.37],
            4: [3, 1, 98.37]
        }, index=["Suction Pressure", "Suction temp", "Condenser In"]),
        pd.DataFrame({
            0: [1, 2, 98.43, 109.31],
            1: [1, 3, 97.64, 109.18],
            2: [1, 4, 96.866667, 109.18],
            3: [1, 5, 96.093333, 109.18],
            4: [1, 6, 95.32, 109.18],
            5: [1, 6, 95.32, 110.20]
        }, index=["Suction Pressure", "Suction temp", "Condenser In", "Condenser Out"]),
        pd.DataFrame({
            0: [61.71, 102.75],
            1: [69.21, 103.19],
            2: [71.35, 103.34],
            3: [68.14, 102.60],
            4: [83.47, 103.26],
            5: [63.14, 103.85]
        }, index=['Suction Pressure', 'Suction temp']),
        pd.DataFrame({
            0: [61.71, 98.30, 55.31, 100.32],
            1: [102.75, 98.30, 71.35, 100.32],
            2: [106.24, 98.30, 103.34, 100.32],
            3: [109.73, 98.30, 106.105, 102.6],
            4: [110.81, 98.30, 108.87, 98.37],
            5: [111.32, 98.30, 109.32, 98.37],
            6: [111.32, 98.30, 109.32, 98.37]
        })
    ]

    return dfs, dfs_expected


def test_csv_reader_NA(tmp_path, setup_csvreader_data, mocker):
    dfs, dfs_expected = setup_csvreader_data
    d = tmp_path / "na_test.csv"
    d.write_text("")
    csv_reader = CSVReader(str(d))
    for input_df, expected_df in zip(dfs, dfs_expected):
        mocker.patch('pandas.read_csv', return_value=input_df)
        result_df = csv_reader.fetch()
        try:
            pd.testing.assert_frame_equal(result_df, expected_df,
                                      check_dtype=False, check_index_type=False)
        except AssertionError as e:
            raise AssertionError(f"Data transformation did not produce the expected output. Expected {expected_df}, instead got {result_df}") from e

def test_csv_reader_columns(tmp_path):
    d = tmp_path / "col_names_test.csv"
    d.write_text("0, 1, 2, 3, 4,LABEL\n1.,1.,1.,1.,1.,hello\n 2, 3, 4, 5, 6,world")
    csv_reader = CSVReader(str(d))
    df = csv_reader.fetch()
    assert df.shape == (2, 5)
    for col in df.columns:
        assert is_float_dtype(df[col])

def test_invalid_csv_path():
    data_path = "invalid/path/to/data.csv"
    csv_reader = CSVReader(data_path)
    with pytest.raises(FileNotFoundError):
        csv_reader.fetch()

def test_empty_csv(tmp_path):
    d = tmp_path / "empty.csv"
    d.write_text("")
    csv_reader = CSVReader(str(d))
    with pytest.raises(ValueError):
        csv_reader.fetch()

def test_improper_csv(tmp_path):
    d = tmp_path / "improper.csv"
    d.write_text('''0,1,2,3,4,5\n,0.,1.,2,3,4,"5''')
    csv_reader = CSVReader(str(d))
    with pytest.raises(ValueError):
        csv_reader.fetch()

def test_float_indices(tmp_path):
    d = tmp_path / "float_indices.csv"
    data_string = """date,0.0,1.0,2.0\n
                    2014,0.25,0.50,0.75\n
                    2015,0.25,0.50,0.75\n
                    2016,0.25,0.50,0.75\n
                    2017,0.25,0.50,0.75\n
                    2018,0.25,0.50,0.75"""
    d.write_text(data_string)
    csv_reader = CSVReader(str(d))
    df = csv_reader.fetch()

    expected_df = pd.DataFrame({
        0: [0.25, 0.50, 0.75],
        1: [0.25, 0.50, 0.75],
        2: [0.25, 0.50, 0.75],
        3: [0.25, 0.50, 0.75],
        4: [0.25, 0.50, 0.75]
    }, index=[0, 1, 2])

    try:
        pd.testing.assert_frame_equal(df, expected_df,
                                    check_dtype=False)
    except AssertionError as e:
        raise AssertionError(f"Data transformation did not produce the expected output. Expected {expected_df}, instead got {df}") from e

@pytest.fixture
def setup_csv_data(mocker):
    # 准备输入的 DataFrame 和期望的输出
    dfs = [
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In'],
            ['6/30/2023 19:01:24', 0, 1, 98.34],
            ['6/30/2023 19:03:04', 0, 3, 98.93],
            ['6/30/2023 19:04:44', 0, 2, 97.90],
            ['6/30/2023 19:06:22', 2, 3, 98.37],
            ['6/30/2023 19:08:03', 3, 1, 98.37]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
            ['6/30/2023 19:09:43', 1, 2, 98.43, 109.31],
            ['6/30/2023 19:11:23', 1, 3, 97.64, 109.18],
            ['6/30/2023 19:13:02', 1, 4, np.nan, 109.18]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp'],
            ['6/30/2023 19:01:24', 61.71, 102.75],
            ['6/30/2023 19:03:04', 69.21, 103.19],
            ['6/30/2023 19:04:44', 71.35, 103.34],
            ['6/30/2023 19:06:22', 68.14, 102.60],
            ['6/30/2023 19:08:03', 83.47, 103.26],
            ['6/30/2023 19:09:43', 63.14, 103.85]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
            ['6/30/2023 19:01:24', 61.71, 102.75, np.nan, 109.73],
            ['6/30/2023 19:03:04', np.nan, 103.19, 98.93, 109.73],
            ['6/30/2023 19:04:44', 71.35, 103.34, np.nan, 108.87],
            ['6/30/2023 19:06:22', np.nan, 102.60, 98.37, np.nan]
        ]),
        pd.DataFrame([
            ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out', 'Additional Column'],
            ['6/30/2023 19:01:24', 61.71, 102.75, 98.34, 109.73, 1],
            ['6/30/2023 19:03:04', 69.21, 103.19, 98.93, 109.73, 2],
            ['6/30/2023 19:04:44', 71.35, 103.34, 97.90, 108.87, 3],
            ['6/30/2023 19:06:22', 68.14, 102.60, 98.37, 109.48, 4],
            ['6/30/2023 19:08:03', 83.47, 103.26, 98.37, 109.44, 5]
        ])
    ]

    dfs_expected = [
        pd.DataFrame({
            0: [0, 1, 98.34],
            1: [0, 3, 98.93],
            2: [0, 2, 97.90],
            3: [2, 3, 98.37],
            4: [3, 1, 98.37]
        }),
        pd.DataFrame({
            0: [1, 2, 98.43, 109.31],
            1: [1, 3, 97.64, 109.18],
            2: [1, 4, 0, 109.18]
        }),
        pd.DataFrame({
            0: [61.71, 102.75],
            1: [69.21, 103.19],
            2: [71.35, 103.34],
            3: [68.14, 102.60],
            4: [83.47, 103.26],
            5: [63.14, 103.85]
        }),
        pd.DataFrame({
            0: [61.71, 102.75, 0, 109.73],
            1: [0, 103.19, 98.93, 109.73],
            2: [71.35, 103.34, 0, 108.87],
            3: [0, 102.6, 98.37, 0]
        }),
        pd.DataFrame({
            0: [61.71, 102.75, 98.34, 109.73, 1],
            1: [69.21, 103.19, 98.93, 109.73, 2],
            2: [71.35, 103.34, 97.9, 108.87, 3],
            3: [68.14, 102.6, 98.37, 109.48, 4],
            4: [83.47, 103.26, 98.37, 109.44, 5]
        })
    ]

    return dfs, dfs_expected


def test_transform_csv(setup_csv_data, mocker):
    dfs, dfs_expected = setup_csv_data
    for i, (input_df, expected_df) in enumerate(zip(dfs, dfs_expected)):
        mocker.patch('pandas.read_csv', return_value=input_df)
        result_df = transform_csv(f'file{i+1}.csv')
        #assert (result_df.iloc[0, :] == range(len(result_df.columns))).all(), "Time sequence conversion failed." Q to ask
        try:
            pd.testing.assert_frame_equal(result_df, expected_df,
                                      check_dtype=False,)
        except AssertionError as e:
            raise AssertionError("Data transformation did not produce the expected output.") from e


def test_transform_csv_folder(mocker):
    """ This test case tests the transform_csv_dataset function with different inputs. """
    
    mocker.patch('os.path.exists', side_effect=lambda path: path == './input_folder')
    mocker.patch('os.listdir', return_value=['file1.csv', 'file2.csv'])
    
    mock_transform_csv = mocker.patch('ltsm.data_reader.csv_reader.transform_csv')
    mock_to_csv = mocker.patch('pandas.DataFrame.to_csv')
    mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mock_transform_csv.return_value = mock_df
    mock_makedirs = mocker.patch('os.makedirs')

    transform_csv_dataset('./input_folder', './output_folder')
    mock_makedirs.assert_called_once_with('./output_folder')
    assert mock_transform_csv.call_count == 2  # check if transform_csv is called twice

    expected_calls = [
        mocker.call(os.path.join('./output_folder', 'file1.csv'), index=False),
        mocker.call(os.path.join('./output_folder', 'file2.csv'), index=False)
    ]
    mock_to_csv.assert_has_calls(expected_calls, any_order=True)