import argparse
from pathlib import Path
import numpy as np


def read_lines(path: Path):
    with open(path, "r") as f:
        return f.readlines()


def write_lines(path: Path, lines):
    with open(path, "w") as f:
        f.writelines(lines)


def reorder_lines(lines, order):
    return [lines[i] for i in order]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly reorder dataset rows after a fixed prefix while keeping "
            "features, xyz-list, and low-level energies aligned."
        )
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--xyz-list", required=True)
    parser.add_argument("--low-level-energies", required=True)
    parser.add_argument("--ccsd-energies", required=True)
    parser.add_argument("--fixed-prefix", type=int, default=972)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-features", required=True)
    parser.add_argument("--out-xyz-list", required=True)
    parser.add_argument("--out-low-level-energies", required=True)
    parser.add_argument("--out-ccsd-energies", required=True)
    parser.add_argument("--out-permutation", required=True)
    args = parser.parse_args()

    features_lines = read_lines(Path(args.features))
    xyz_lines = read_lines(Path(args.xyz_list))
    low_lines = read_lines(Path(args.low_level_energies))
    ccsd_lines = read_lines(Path(args.ccsd_energies))

    n_features = len(features_lines)
    n_xyz = len(xyz_lines)
    n_low = len(low_lines)
    n_ccsd = len(ccsd_lines)

    print(f"Number of feature rows:       {n_features}")
    print(f"Number of xyz-list rows:      {n_xyz}")
    print(f"Number of low-level rows:     {n_low}")
    print(f"Number of CCSD rows:          {n_ccsd}")
    print(f"Fixed prefix:                 {args.fixed_prefix}")

    if not (n_features == n_xyz == n_low):
        raise ValueError(
            "features, xyz-list, and low-level energies must have the same number of rows."
        )

    if n_ccsd < args.fixed_prefix:
        raise ValueError(
            f"CCSD file has {n_ccsd} rows but fixed prefix is {args.fixed_prefix}."
        )

    if args.fixed_prefix > n_features:
        raise ValueError("Fixed prefix is larger than the dataset.")

    rng = np.random.default_rng(args.seed)

    fixed = np.arange(args.fixed_prefix)
    tail = np.arange(args.fixed_prefix, n_features)
    shuffled_tail = rng.permutation(tail)

    order = np.concatenate([fixed, shuffled_tail])

    # Reorder only files with full length.
    write_lines(Path(args.out_features), reorder_lines(features_lines, order))
    write_lines(Path(args.out_xyz_list), reorder_lines(xyz_lines, order))
    write_lines(Path(args.out_low_level_energies), reorder_lines(low_lines, order))

    # CCSD has known labels only. Because the first fixed-prefix rows remain unchanged,
    # we can simply copy the CCSD file unchanged.
    write_lines(Path(args.out_ccsd_energies), ccsd_lines)

    np.savetxt(args.out_permutation, order, fmt="%d")

    print("\nWrote:")
    print(f"  {args.out_features}")
    print(f"  {args.out_xyz_list}")
    print(f"  {args.out_low_level_energies}")
    print(f"  {args.out_ccsd_energies}")
    print(f"  {args.out_permutation}")
    print("\nImportant: CCSD rows were copied unchanged because rows 0-971 were kept fixed.")


if __name__ == "__main__":
    main()