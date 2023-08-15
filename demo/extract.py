import numpy as np
from patchify import patchify
from scipy.io import loadmat
import re
import pandas as pd


def extract_license_UFPR_plate_number(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        input_text = file.read()

    # Use regular expression to find the line containing the plate number
    pattern = r'plate: (\S+)'
    match = re.search(pattern, input_text)

    if match:
        plate_number = match.group(1)
        return plate_number
    else:
        return None
    
def extract_tagnumber_RBNR(file_path):
    """
    inputs file path
    return the number
    """
    return loadmat(file_path)['number']


def extract_last_word_from_file(file_path):
    """
    Takes input as the filepath
    return the last word if line which contain the jersy id
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    last_words = []
    for line in lines:
        words = line.strip().split(',')
        last_word = words[-1].strip()
        last_words.append(last_word)

    return last_words

def extract_plate_value(file_path):
    """
    takes filepath as input
    returns the numberplate value
    """
    # Open the text file
    with open(file_path, 'r') as file:
        # Read the file contents
        content = file.read()

        # Use regex to find the plate value
        plate_match = re.search(r'type: car\nplate: ([A-Z0-9]+)', content)

        if plate_match:
            # Extract the plate value
            plate_value = plate_match.group(1)
            return plate_value
        else:
            return ""