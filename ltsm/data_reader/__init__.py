from ltsm.data_reader.monash_reader import MonashReader
from ltsm.data_reader.csv_reader import CSVReader
reader_dict = {}

def register_reader(module):
    """
    Registers a BaseReader module into the reader dictionary.

    Args:
        module: A Python module or class that implements a BaseReader.
        module_name (str): The key name for the module in the reader dictionary.

    Raises:
        AssertionError: If a reader with the same name is already registered
    """
    assert module.module_id not in reader_dict, f"Reader {module.module_id} already registered"
    reader_dict[module.module_id] = module

register_reader(MonashReader)
register_reader(CSVReader)

__all__ = {
    register_reader,
    MonashReader,
    CSVReader
}