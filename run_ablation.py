"""Paper Table 2: six cumulative ablations; fixed .5 threshold, no TTA."""
import argparse
from pathlib import Path
import subprocess
import sys

CONFIGS = [
    ('baseline',['--no-stage0','--no-as-mscb','--no-rmsab','--no-aaf','--no-edge','--no-cldice']),
    ('cldice',['--no-stage0','--no-as-mscb','--no-rmsab','--no-aaf','--no-edge']),
    ('edge',['--no-stage0','--no-as-mscb','--no-rmsab','--no-aaf']),
    ('aaf',['--no-stage0','--no-as-mscb','--no-rmsab']),
    ('rmsab_as_mscb',['--no-stage0']),
    ('beamnet',[]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--execute',action='store_true',help='Otherwise print commands only')
    parser.add_argument('--no-pretrain',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_root = str(Path(args.data_root).resolve())
    for index,(name,flags) in enumerate(CONFIGS,1):
        run_id = 'ablation_{}_{}'.format(index,name)
        train = [sys.executable,str(root/'train.py'),'--data-root',data_root,'--run-id',run_id]+flags
        if args.no_pretrain:
            train.append('--no-pretrain')
        test = [sys.executable,str(root/'test_final.py'),'--data-root',data_root,
                '--checkpoint','runs/'+run_id+'/best.pth','--output','results/'+run_id]
        test += [f for f in flags if f != '--no-cldice'] + ['--no-tta', '--fixed-threshold']
        for command in (train,test):
            print(subprocess.list2cmdline(command),flush=True)
            if args.execute:
                subprocess.run(command,cwd=str(root),check=True)


if __name__ == '__main__':
    main()
