import unittest
import pandas as pd
from io import StringIO

class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        # Create a simulated CSV input data
        self.input_csv = StringIO(
            """Updated Time,Suction Pressure,Suction temperature,Condenser Inlet Temperature,Condenser Outlet Temperature,Liquid temperature,Liquid Pressure,Compressor current,Condensing Fan Current,Top Shell Temperature,Discharge Temperature,Bottom Temperature,Motor Temperature
            6/30/2023 19:01:24,61.712231658240015,102.75,98.340625,109.73125,100.84,363.4015032,9.9,0.6,58.22,26.27,118.6609375,118.8859375
            6/30/2023 19:03:04,69.21224676096001,103.19,98.93125,109.73125,100.84,364.13170107648006,9.86,0.6,58.89,26.4,118.365625,118.6046875
            """
        )

        # Expected format of converted data
        self.expected_df = pd.DataFrame({
            0: [0, 1],
            1: [61.712231658240015, 69.21224676096001],
            2: [102.75, 103.19],
            3: [98.340625, 98.93125],
            4: [109.73125, 109.73125],
            5: [100.84, 100.84],
            6: [363.4015032, 364.13170107648006],
            7: [9.9, 9.86],
            8: [0.6, 0.6],
            9: [58.22, 58.89],
            10: [26.27, 26.4],
            11: [118.6609375, 118.365625],
            12: [118.8859375, 118.6046875]
        })

    def test_data_transformation(self):
        # Read CSV data as a DataFrame
        input_df = pd.read_csv(self.input_csv, parse_dates=['Updated Time'])

        # Execute data conversion function
        transformed_df = transform_data(input_df)

        # Verify that the time column has been successfully converted to the 0, 1, 2... format
        self.assertTrue((transformed_df.iloc[0, :] == range(len(transformed_df.columns))).all(),
                        "Time sequence conversion failed.")

        # Verify that the converted data structure meets expectations
        pd.testing.assert_frame_equal(
            transformed_df.iloc[1:, :].reset_index(drop=True),
            self.expected_df.reset_index(drop=True),
            check_dtype=False,
            err_msg="Data transformation did not produce the expected output."
        )

if __name__ == '__main__':
    unittest.main()
    # Step 1
    # Step 2
