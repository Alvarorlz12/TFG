import csv

def write_csv(file_path, data, header=None):
    """
    Write data to a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    data : list of lists
        Data to be written to the CSV file. Each inner list represents a row.
    header : list, optional
        Header for the CSV file. If provided, it will be written as the first 
        row.
    """
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        if header is not None:
            writer.writerow(header)
        writer.writerows(data)

def write_csv_from_dict(file_path, data_dict, fields=None, header=None):
    """
    Write data from a dictionary of dictionaries to a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    data_dict : dict of dicts
        Data to be written to the CSV file. Each inner dictionary represents a 
        row, with keys as column names and values as cell values. The outer
        dictionary's keys are used as the row identifiers (e.g., patient IDs).
    fields : list, optional
        List of fields to include in the CSV file. If None, all fields from 
        the first dictionary will be used.
    header : list, optional
        Header for the CSV file. If provided, it will be written as the first 
        row. If None, the keys of the first dictionary will be used as the 
        header.
    """
    if fields is None:
        fields = list(data_dict.values())[0].keys()
    
    if header is None:
        header = list(data_dict.values())[0].keys()

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for key, value in data_dict.items():
            row = [key] + [value[field] for field in fields]
            writer.writerow(row)