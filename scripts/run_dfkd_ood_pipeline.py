#!/usr/bin/env python3
"""Leakage-safe DFKD/OOD-KD continuation: no CIFAR-10 train/validation access.

The fixed protocol is manual rather than AutoML: balanced Gaussian baseline,
balanced SVHN arbitrary transfer, then DeepInversion-style synthetic + OOD
curriculum. The optional final test command is the sole target-test unlock.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
 sys.path.insert(0,str(ROOT))
OUT=ROOT/'dfkd_ood_2026'; TRANSFER=OUT/'transfer'; CKPT=OUT/'checkpoints'; PLOTS=OUT/'plots'; RESULTS=OUT/'results'
MEAN=(.4914,.4822,.4465); STD=(.2470,.2435,.2616); CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def args():
 p=argparse.ArgumentParser(); p.add_argument('--locked-final-test',action='store_true'); p.add_argument('--seed',type=int,default=2026); p.add_argument('--epochs',type=int,default=35); return p.parse_args()
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark=True
def norm(x):
 m=torch.tensor(MEAN,device=x.device).view(1,3,1,1); s=torch.tensor(STD,device=x.device).view(1,3,1,1); return (x-m)/s
def denorm(x):
 m=torch.tensor(MEAN,device=x.device).view(1,3,1,1); s=torch.tensor(STD,device=x.device).view(1,3,1,1); return (x*s+m).clamp(0,1)
def save(obj,path): path.parent.mkdir(parents=True,exist_ok=True); torch.save(obj,path)
def jdump(x,path): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,indent=2)+'\n')
def checksum(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

def load_teacher(device):
 from src.models.resnet_cifar import resnet34_cifar
 path=ROOT/'resnet34_cifar10_fp32_best.pth'; ck=torch.load(path,map_location=device,weights_only=False)
 assert ck['arch']=='ResNet34-CIFAR'; model=resnet34_cifar().to(device); model.load_state_dict(ck['model_state_dict']); model.eval()
 for p in model.parameters(): p.requires_grad_(False)
 return model, path

class BNHook:
 def __init__(self,m): self.loss=torch.tensor(0.); self.h=m.register_forward_hook(self.fn)
 def fn(self,m,inp,out):
  x=inp[0]; mean=x.mean((0,2,3)); var=x.var((0,2,3),unbiased=False)
  self.loss=F.mse_loss(mean,m.running_mean)+F.mse_loss(var,m.running_var)
 def close(self): self.h.remove()

def teacher_stats(teacher,x):
 with torch.no_grad():
  z=teacher(x); p=z.softmax(1); top=p.argmax(1); conf=p.max(1).values; ent=-(p*p.clamp_min(1e-8).log()).sum(1); margin=p.topk(2,1).values.diff(dim=1).abs().squeeze(1)
 return z.cpu(),top.cpu(),conf.cpu(),ent.cpu(),margin.cpu()
def teacher_stats_batched(teacher,x,batch_size=256):
 rows=[teacher_stats(teacher,x[i:i+batch_size]) for i in range(0,len(x),batch_size)]
 return tuple(torch.cat(parts) for parts in zip(*rows))

def balanced_select(x,z,top,conf,per_class=384):
 selected=[]
 for c in range(10):
  ids=(top==c).nonzero().flatten(); ranked=ids[conf[ids].argsort(descending=True)]; selected.append(ranked[:min(per_class,len(ranked))])
 ids=torch.cat(selected); return x[ids],z[ids],top[ids],conf[ids]

def svhn_transfer(teacher,device,seed):
 path=TRANSFER/'svhn_balanced.pt'
 if path.exists(): return torch.load(path,weights_only=False)
 tf=transforms.Compose([transforms.ToTensor()])
 ds=datasets.SVHN(str(OUT/'external'),split='train',transform=tf,download=True) # labels deliberately ignored
 g=torch.Generator().manual_seed(seed); ids=torch.randperm(len(ds),generator=g)[:12000].tolist()
 loader=DataLoader(torch.utils.data.Subset(ds,ids),batch_size=256,shuffle=False,num_workers=4,pin_memory=True)
 xs=[]
 for images,_ in loader: xs.append(images)
 x=torch.cat(xs); z,top,conf,ent,margin=teacher_stats_batched(teacher,norm(x.to(device)))
 x,z,top,conf=balanced_select(x,z,top,conf); data={'images':x,'logits':z,'pseudo_class':top,'confidence':conf,'source':'SVHN train split (unlabeled)','source_url':'http://ufldl.stanford.edu/housenumbers/','license':'SVHN academic dataset; labels ignored','candidate_count':12000,'selection':'teacher-pseudo-class balanced, top confidence within each class','sha256':checksum(x)}; save(data,path); return data

def gaussian_transfer(teacher,device,seed):
 path=TRANSFER/'gaussian_balanced.pt'
 if path.exists(): return torch.load(path,weights_only=False)
 g=torch.Generator().manual_seed(seed); x=torch.randn((16000,3,32,32),generator=g).mul(.25).add(.5).clamp(0,1); z,top,conf,ent,margin=teacher_stats_batched(teacher,norm(x.to(device))); x,z,top,conf=balanced_select(x,z,top,conf,128); data={'images':x,'logits':z,'pseudo_class':top,'confidence':conf,'source':'Gaussian noise','selection':'teacher-pseudo-class balanced','candidate_count':16000,'sha256':checksum(x)}; save(data,path); return data

def synthesize(teacher,device,seed):
 path=TRANSFER/'deepinversion_balanced.pt'
 if path.exists(): return torch.load(path,weights_only=False)
 hooks=[BNHook(m) for m in teacher.modules() if isinstance(m,nn.BatchNorm2d)]; all_x=[]; all_z=[]; logs=[]
 for block in range(10):
  target=torch.arange(10,device=device).repeat_interleave(26); x=torch.randn((260,3,32,32),device=device).mul(.15).add(.5).requires_grad_(); opt=torch.optim.Adam([x],lr=.12,betas=(.5,.9))
  for it in range(160):
   opt.zero_grad(); logits=teacher(norm(x.clamp(0,1))); ce=F.cross_entropy(logits,target); bn=torch.stack([h.loss for h in hooks]).sum(); tv=(x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()+(x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean(); diversity=-x.flatten(1).std(1).mean(); loss=ce+3.0*bn+1e-4*tv+1e-3*diversity; loss.backward(); opt.step()
   with torch.no_grad(): x.clamp_(0,1)
  z,top,conf,ent,margin=teacher_stats(teacher,norm(x.detach())); all_x.append(x.detach().cpu()); all_z.append(z); logs.append({'block':block,'teacher_confidence':float(conf.mean()),'entropy':float(ent.mean()),'bn_loss':float(bn.detach())})
 for h in hooks:h.close()
 x=torch.cat(all_x); z=torch.cat(all_z); _,top,conf,_,_=teacher_stats_batched(teacher,norm(x.to(device))); data={'images':x,'logits':z,'pseudo_class':top,'confidence':conf,'source':'DeepInversion synthetic teacher-guided samples','selection':'fixed balanced targets + teacher BN running-statistics match + total-variation/image diversity priors','synthesis_steps':160,'samples':len(x),'logs':logs,'sha256':checksum(x)}; save(data,path); return data

def make_student(device):
 from src.models.resnet_cifar import resnet18_cifar
 from src.quant.ternary import QuantConfig,convert_to_ternary
 model=resnet18_cifar().to(device); convert_to_ternary(model,QuantConfig(threshold='twn',scale='derived',scope='per_channel',ste='clipped')); return model
def kd(student,z,t=2.): return F.kl_div(F.log_softmax(student/t,1),F.softmax(z/t,1),reduction='batchmean')*(t*t)
def train(name,train_data,holdout_data,device,seed,epochs,source_mix=None):
 from src.quant.ternary import ternary_parameter_groups
 path=CKPT/f'{name}_best.pth'; hist_path=RESULTS/f'{name}_history.json'
 if path.exists(): return torch.load(path,map_location='cpu',weights_only=False)
 model=make_student(device); opt=torch.optim.SGD(ternary_parameter_groups(model,1e-4),lr=.08,momentum=.9,nesterov=True); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best=1e9; hist=[]
 x,z=train_data['images'],train_data['logits']; hx,hz=holdout_data['images'],holdout_data['logits']; ds=TensorDataset(x,z); loader=DataLoader(ds,batch_size=128,shuffle=True,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(seed))
 for ep in range(epochs):
  model.train(); total=agree=n=0
  for a,b in loader:
   a,b=a.to(device),b.to(device); opt.zero_grad(set_to_none=True); out=model(norm(a)); loss=kd(out,b); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=loss.item()*len(a); agree+=(out.argmax(1)==b.argmax(1)).sum().item();n+=len(a)
  model.eval(); vl=va=vn=0
  with torch.no_grad():
   for a,b in DataLoader(TensorDataset(hx,hz),batch_size=256):
    out=model(norm(a.to(device))); l=kd(out,b.to(device));vl+=l.item()*len(a);va+=(out.argmax(1).cpu()==b.argmax(1)).sum().item();vn+=len(a)
  row={'epoch':ep+1,'train_kd_loss':total/n,'train_agreement':agree/n,'transfer_holdout_kd_loss':vl/vn,'transfer_holdout_agreement':va/vn,'lr':opt.param_groups[0]['lr']};hist.append(row);sch.step()
  if row['transfer_holdout_kd_loss']<best:
   best=row['transfer_holdout_kd_loss']; torch.save({'model_state_dict':model.state_dict(),'arch':'ResNet18-CIFAR-ternary','quant_config':{'threshold':'twn','scale':'derived','symmetric':True,'scope':'per_channel','t':.05,'twn_factor':.7,'ste':'clipped','clip':1.},'epoch':ep+1,'selection_metric':'external/synthetic transfer holdout KD loss','best_transfer_kd_loss':best,'seed':seed,'data_policy':'No CIFAR-10 train/validation data used','transfer_sources':source_mix or [train_data['source']]},path)
 jdump({'name':name,'epochs':epochs,'history':hist,'selection':'minimum held-out transfer KD loss; no target data'},hist_path); return torch.load(path,map_location='cpu',weights_only=False)

def locked_test(path,device):
 # This is deliberately the only target-dataset construction in this file.
 from src.data import get_test_loader
 from src.evaluation.verify_ternary import verify_model
 ck=torch.load(path,map_location=device,weights_only=False); model=make_student(device);model.load_state_dict(ck['model_state_dict']); verify=verify_model(model,verbose=False); loader=get_test_loader(ROOT/'data',256,4,final_evaluation=True); cm=torch.zeros(10,10,dtype=torch.int64);model.eval()
 with torch.no_grad():
  for x,y in loader:
   pred=model(x.to(device)).argmax(1).cpu(); cm.index_put_((y,pred),torch.ones_like(y,dtype=torch.int64),accumulate=True)
 per=cm.diag().float()/cm.sum(1); return {'checkpoint':str(path.relative_to(ROOT)),'test_accuracy':float(cm.diag().sum()/cm.sum()),'per_class_accuracy':{CLASSES[i]:float(per[i]) for i in range(10)},'confusion_matrix':cm.tolist(),'ternary_verification':verify,'protocol':'one locked post-freeze test evaluation'}

def main():
 a=args();set_seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');assert device.type=='cuda'; OUT.mkdir(exist_ok=True); [p.mkdir(parents=True,exist_ok=True) for p in [TRANSFER,CKPT,PLOTS,RESULTS]]
 teacher,_=load_teacher(device); gaussian=gaussian_transfer(teacher,device,a.seed); svhn=svhn_transfer(teacher,device,a.seed); synth=synthesize(teacher,device,a.seed)
 # Fixed, predeclared selection: use 10% source-held-out data only; never CIFAR validation.
 def split(d): n=len(d['images']);cut=int(.9*n);return ({**d,'images':d['images'][:cut],'logits':d['logits'][:cut]},{**d,'images':d['images'][cut:],'logits':d['logits'][cut:]})
 gn,gh=split(gaussian); train('gaussian_balanced_baseline',gn,gh,device,a.seed,5)
 sn,sh=split(synth); on,oh=split(svhn); mix={'images':torch.cat([sn['images'],on['images']]),'logits':torch.cat([sn['logits'],on['logits']]),'source':'DeepInversion+balanced SVHN curriculum'}; hold={'images':torch.cat([sh['images'],oh['images']]),'logits':torch.cat([sh['logits'],oh['logits']]),'source':'heldout synthetic+SVHN'}; final=train('dfkd_di_svhn_ternary',mix,hold,device,a.seed,a.epochs,['DeepInversion BN-stat synthetic','SVHN unlabeled teacher-balanced'])
 # diagnostics + provenance
 for name,d in [('gaussian',gaussian),('svhn',svhn),('deepinversion',synth)]:
  c=Counter(d['pseudo_class'].tolist()); fig,ax=plt.subplots(1,2,figsize=(10,3.5));ax[0].bar(range(10),[c[i] for i in range(10)]);ax[0].set(title=f'{name}: teacher pseudo-class balance',xlabel='pseudo-class',ylabel='samples');ax[1].hist(d['confidence'].numpy(),bins=25);ax[1].set(title=f'{name}: teacher confidence',xlabel='confidence');fig.tight_layout();fig.savefig(PLOTS/f'{name}_transfer_diagnostics.png',dpi=160);plt.close(fig)
 jdump({'objective':'DFKD/OOD strict ternary QAT without CIFAR-10 training or validation data','literature':['Nayak et al. 2020 arbitrary transfer sets, arXiv:2011.09113','Yin et al. 2020 DeepInversion, CVPR','Choi et al. 2020 data-free network quantization, CVPRW','Liu et al. 2024 small-scale DFKD, CVPR'],'fixed_protocol':{'baseline':'balanced Gaussian, 5 epochs','final':'balanced SVHN + DeepInversion BN-stat synthetic, 35 epochs','selection':'transfer-held-out KD loss only'},'transfer_sources':[gaussian['source'],svhn['source'],synth['source']],'provenance':{'svhn_url':'http://ufldl.stanford.edu/housenumbers/','labels':'ignored'},'final_checkpoint':str((CKPT/'dfkd_di_svhn_ternary_best.pth').relative_to(ROOT))},RESULTS/'protocol.json')
 if a.locked_final_test:
  out=RESULTS/'locked_final_test.json';
  if out.exists():
   raise FileExistsError('locked result exists')
  jdump(locked_test(CKPT/'dfkd_di_svhn_ternary_best.pth',device),out)
 print('DFKD pipeline complete; test=',a.locked_final_test)
if __name__=='__main__':main()
