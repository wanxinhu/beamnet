"""Evaluate a matching checkpoint; calibrate thresholds on validation only."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.data import records
from utils.evaluation import collect, summarize, select_threshold
from utils.runtime import common_parser, setup, build_model, load_weights


def main():
    parser = common_parser('BEAM-Net evaluation')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--no-tta',action='store_true',help='Disable the paper test-time augmentation')
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--fixed-threshold',action='store_true',help='Use --threshold instead of validation calibration')
    parser.add_argument('--region',choices=['legacy','fov'],default='legacy')
    parser.add_argument('--raw-probabilities',action='store_true',help='Disable historical per-image min-max normalization')
    parser.add_argument('--output',default='results')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('threshold must be between zero and one')
    device = setup(args)
    model = build_model(args,device)
    load_weights(model,args.checkpoint,device)
    threshold = args.threshold
    tta = not args.no_tta
    search_threshold = not args.fixed_threshold
    if search_threshold:
        validation = collect(model,records(args.data_root,args.dataset,'val'),device,tta,not args.raw_probabilities)
        threshold = select_threshold(validation,args.region)
    predictions = collect(model,records(args.data_root,args.dataset,args.split),device,tta,not args.raw_probabilities)
    rows,scores = summarize(predictions,threshold,args.region)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    for key,prob,_,fov in predictions:
        Image.fromarray(((prob >= threshold) & fov).astype(np.uint8)*255).save(output/(key+'.png'))
        np.save(output/(key+'_prob.npy'),prob)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(dataset=args.dataset,split=args.split,threshold=threshold,
                  threshold_source='validation' if search_threshold else 'fixed',tta=tta,
                  region=args.region,normalization=not args.raw_probabilities,count=len(rows),mean=scores)
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
