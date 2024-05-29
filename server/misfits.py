import argparse
import numpy as np
import os
import pandas as pd
import re
import scipy.interpolate as sp_int


def read_receiver(filename):
    with open(filename) as receiver_file:
        receiver_file.readline()
        quantity_line = receiver_file.readline().strip()
        header_length = 2
        while receiver_file.readline()[0] == "#":
            header_length += 1
    quantity_line = quantity_line.replace('"', "")
    quantities = quantity_line[12:].split(",")
    receiver_df = pd.read_csv(filename, skiprows=header_length, header=None, sep="\s+")
    receiver_df.columns = quantities
    return receiver_df


def receiver_difference(simulation, reference):
    assert (simulation.columns == reference.columns).all()

    misfit = 0
    time = reference["Time"]
    for q in ["v1", "v2", "v3"]:
        interpolator = sp_int.interp1d(simulation["Time"], simulation[q])
        q_interpolated = interpolator(time)
        q = reference[q]
        diff = np.sqrt(np.trapz((q - q_interpolated) ** 2, time))
        norm = np.sqrt(np.trapz(q**2, time))
        misfit += diff / norm / 3.0
    return misfit


def find_receiver(directory, prefix, number):
    receiver_re = re.compile(f"{prefix}-receiver-{number:05d}-(\d)+.dat")
    for fn in os.listdir(directory):
        if receiver_re.match(fn):
            return os.path.join(directory, fn)


def misfit(directory_simulation, directory_reference, prefix, number):
    receiver_simulation = read_receiver(
        find_receiver(directory_simulation, prefix, number)
    )
    receiver_reference = read_receiver(
        find_receiver(directory_reference, prefix, number)
    )
    return receiver_difference(receiver_simulation, receiver_reference)


if __name__ == "__main__":
    directory_simulation = "simulation"
    directory_reference = "reference"
    prefix = "tpv5"

    for i in range(1, 6):
        receiver_simulation = read_receiver(
            find_receiver(directory_simulation, prefix, i)
        )
        receiver_reference = read_receiver(
            find_receiver(directory_reference, prefix, i)
        )
        print(receiver_difference(receiver_simulation, receiver_reference))
