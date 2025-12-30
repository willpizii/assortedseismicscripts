from asnlib.preprocess.rotate import (
    get_az_baz,
    pairwise_rotation_matrix,
    find_component_files,
    load_components,
    rotate_rr_tt,
)

from asnlib.io.write import write_with_template

import os
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def rotate_ccf(station1, station2, CC_dir, RR_dir, TT_dir):
    """
    Rotates cross correlations from pairwise NN,EE,EN,NE to RR and TT.
    For now, works with msnoise standard file structure.
    Assumes components per station are constant through stack period.
    
    Parameters:
    station1: msnoise Station object, or dict with X, Y, net and sta
    station2: msnoise Station object, or dict with X, Y, net and sta
    CC_dir: input parent dir with EE, EN, NN and NE cross correlations
    RR_dir: output dir for RR data
    TT_dir: output dir for TT data

    Returns:
    None
    """

    # handle dicts or objects
    get = lambda s, k: getattr(s, k, None) or s[k]

    # ignore autocorrelations
    if get(station1, "sta") == get(station2, "sta"):
        return
    
    # station-pair identifier
    pattern = f"{get(station1,'net')}_{get(station1,'sta')}_{get(station2,'net')}_{get(station2,'sta')}"
    
    # azimuth and backazimuth
    az, baz = get_az_baz(get(station1,'X'), get(station1,'Y'),
                          get(station2,'X'), get(station2,'Y'))

    # rotation operator
    rot_matrix = pairwise_rotation_matrix(az, baz)

    # locate component files
    files = find_component_files(CC_dir, pattern)
    if files is None:
        return

    # iterate over time slices
    for EE_path, EN_path, NN_path, NE_path in zip(
        files["EE"], files["EN"], files["NN"], files["NE"]
    ):
        # load data
        EE, EN, NN, NE = load_components(
            EE_path, EN_path, NN_path, NE_path
        )

        # rotate
        rot = rotate_rr_tt(EE, EN, NN, NE, rot_matrix)
        TT = rot[0]
        RR = rot[1]

        # write outputs using EE as template
        write_with_template(
            EE_path, TT, TT_dir, pattern, EE_path
        )
        write_with_template(
            EE_path, RR, RR_dir, pattern, EE_path
        )

def station_to_dict(station):
    """
    Turns an msnoise station object into a dictionary
    
    station: msnoise station object
    """
    return {"X": station.X, "Y": station.Y, "sta": station.sta, "net": station.net}

def _pair_worker(pair_dicts, CC_dir, RR_dir, TT_dir):
    s1, s2 = pair_dicts
    rotate_ccf(s1, s2, CC_dir, RR_dir, TT_dir)

def _rotate_pairs(station_pairs, CC_dir, RR_dir, TT_dir, max_processes=10):
    os.makedirs(RR_dir, exist_ok=True)
    os.makedirs(TT_dir, exist_ok=True)

    station_pairs_dicts = [(station_to_dict(s1), station_to_dict(s2)) for s1, s2 in station_pairs]

    with ProcessPoolExecutor(max_workers=max_processes) as exe:
        futures = [exe.submit(_pair_worker, pair, CC_dir, RR_dir, TT_dir) for pair in station_pairs_dicts]
        for _ in tqdm.tqdm(as_completed(futures), total=len(futures)):
            pass

def main(cli_args=None):
    import argparse
    from msnoise.api import connect, get_station_pairs

    parser = argparse.ArgumentParser(description="Rotate CCFs in parallel")
    parser.add_argument("CC_dir", help="Directory containing CC stacks")
    parser.add_argument("msn_dir", help="MSNoise base directory")
    parser.add_argument("-t", "--processes", type=int, default=10, help="Number of parallel processes")
    args = parser.parse_args(cli_args)  # <-- use cli_args if provided

    os.chdir(args.msn_dir)
    session = connect()
    station_pairs = get_station_pairs(session, used=True)

    RR_dir = os.path.join(args.CC_dir, "RRPROC")
    TT_dir = os.path.join(args.CC_dir, "TTPROC")

    _rotate_pairs(station_pairs, args.CC_dir, RR_dir, TT_dir, max_processes=args.processes)
