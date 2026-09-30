"""Train BEAM-Net using the paper optimizer and deep supervision."""
import json
import time
from pathlib import Path
from datetime import datetime
import torch
from torch.amp import autocast, GradScaler
from utils.data import records, training_loader
from utils.evaluation import collect, summarize
from utils.losses import combined_vessel_loss, adjust_lr_with_warmup
from utils.runtime import common_parser, setup, build_model, model_config, safe_run_id


def supervised_loss(outputs, target, use_cldice=True):
    # Model order p4, p3, p2, p1, final; paper equation (7).
    return sum(w * combined_vessel_loss(p.float(), target.float(), use_cldice)
               for w, p in zip([.2, .2, .3, .4, 1.], outputs))


def main():
    parser = common_parser('BEAM-Net training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--min-lr', type=float, default=.000001)
    parser.add_argument('--clip', type=float, default=.5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-pretrain', action='store_true')
    parser.add_argument('--no-cldice', action='store_true')
    parser.add_argument('--output', default='runs')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke test only; 0 runs all steps')
    parser.add_argument('--max-val-images', type=int, default=0, help='Smoke test only; 0 uses full validation')
    args = parser.parse_args()
    if args.epochs <= 0 or min(args.warmup_epochs,args.max_steps,args.max_val_images) < 0:
        parser.error('Epochs must be positive; other counts must be nonnegative')
    device = setup(args)
    splits = {split: records(args.data_root,args.dataset,split) for split in ('train','val','test')}
    loader = training_loader(splits['train'],args.batch_size,args.crop_size,args.workers,args.seed)
    run_id = safe_run_id(args.run_id or args.dataset+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output = Path(args.output)/run_id
    output.mkdir(parents=True, exist_ok=False)
    settings = vars(args).copy()
    settings.pop('data_root')
    settings['model'] = model_config(args)
    settings['split_ids'] = {k:[r['id'] for r in v] for k,v in splits.items()}
    settings['loss_weights'] = [.2,.7,.1]
    settings['supervision_weights_p4_to_final'] = [.2,.2,.3,.4,1.]
    settings['amp_enabled'] = device.type == 'cuda'
    (output/'config.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    model = build_model(args,device,pretrained=not args.no_pretrain)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler = GradScaler('cuda',enabled=device.type=='cuda')
    best, start = -1., time.time()
    for epoch in range(1,args.epochs+1):
        lr = adjust_lr_with_warmup(optimizer,epoch,args.epochs,args.warmup_epochs,args.lr,args.min_lr)
        model.train()
        losses = []
        for step,(images,masks) in enumerate(loader,1):
            images,masks = images.to(device),masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type,enabled=device.type=='cuda'):
                outputs = model(images)
                loss = supervised_loss(outputs,masks,not args.no_cldice)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.clamp_(-args.clip,args.clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if args.max_steps and step >= args.max_steps:
                break
        items = splits['val'][:args.max_val_images] if args.max_val_images else splits['val']
        _,scores = summarize(collect(model,items,device),region='legacy')
        torch.save(model.state_dict(),output/'last.pth')
        if scores['F1'] > best:
            best = scores['F1']
            torch.save(model.state_dict(),output/'best.pth')
        row = {'epoch':epoch,'lr':lr,'loss':sum(losses)/len(losses),'val':scores,'best_val_F1':best}
        with (output/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    print('Complete in {:.1f}s. Checkpoint: {}'.format(time.time()-start,output/'best.pth'))


if __name__ == '__main__':
    main()
