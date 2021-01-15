# Authors: Giles Strong, Luca Masserano
# Modified from: https://github.com/GilesStrong/muon_regression

# import uproot
import ROOT
import argparse
from pathlib import Path
import os


def file_is_valid(filename: str, tree: str = 'B4') -> bool:
    try:
        # uproot.open(filename)[tree]
        file = ROOT.TFile.Open(filename)
        t = file.Get(tree)
        if t.GetEntries() < 1:  # corrupted if there are no entries
            raise ValueError("")
    except Exception as e:  # corrupted if another exception is raised
        print(e)
        return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Loop through all ROOT files in a specified directory and produce 
                                    use_files.txt, which lists all the valid (uncorrupted) files, 
                                    and notuse_files.txt (guess what it is). """)

    parser.add_argument("--inputdir", required=False, action="store", type=str, default=Path.cwd(),
                        help="Input directory")
    parser.add_argument("--outputdir", required=False, action="store", type=str, default=Path.cwd(),
                        help="Output directory")
    parser.add_argument("--tree", dest="tree", action="store", type=str, help="Tree name", default='B4')

    args = parser.parse_args()
    inputdir = Path(args.inputdir)
    outputdir = Path(args.outputdir)

    os.system(f'rm {outputdir}/use_files.txt')
    os.system(f'rm {outputdir}/notuse_files.txt')
    with open(outputdir / 'use_files.txt', 'w') as fileoutuse, \
            open(outputdir / 'notuse_files.txt', 'w') as fileoutnotuse:
        for root_file in inputdir.glob('*.root'):
            print(f'Checking {root_file}')
            if file_is_valid(str(root_file), args.tree):
                fileoutuse.write(f'{root_file}\n')
            else:
                fileoutnotuse.write(f'{root_file}\n')
