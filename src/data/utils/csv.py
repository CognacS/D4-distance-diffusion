from typing import List, Tuple

from collections import OrderedDict
import csv


def float_converter(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return x

def read_csv_with_header_and_index(csv_path: str, **csvparams) -> Tuple[List[str], List[str], List[OrderedDict]]:
    """ Read a csv file and return the table, the columns and the index.
    """

    with open(csv_path, 'r') as f:
        reader = csv.reader(f, **csvparams)
        # read header (first row)
        header = next(reader)[1:]
        # read rows from file, and pair them as a list of (id, row data) tuples
        ids_rows = [[row[0], row[1:]] for row in reader]

    # get ids
    ids = [id_row[0] for id_row in ids_rows]

    # get data as a list of rows, with each entry named from the header
    data = [OrderedDict([(k, float_converter(v)) for k, v in zip(header, id_row[1])]) for id_row in ids_rows]

    return header, ids, data