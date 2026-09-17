"""Atomic, verified full optimizer checkpoints; only prune committed siblings."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def complete(root):
    root = Path(root).resolve()
    if not root.exists(): return []
    paths = []
    for p in root.iterdir():
        if p.is_symlink() or not p.is_dir() or not re.fullmatch(r'checkpoint-\d{8}',p.name): continue
        if all((p/n).is_file() and not (p/n).is_symlink() and (p/n).stat().st_size
               for n in ('COMPLETE','state.pt','adapter_config.json','adapter_model.safetensors')):
            paths.append(p)
    return sorted(paths)


def verify(path):
    path = Path(path)
    hashes = json.loads((path/'COMPLETE').read_text())
    if set(hashes) != {'state.pt','adapter_config.json','adapter_model.safetensors'}:
        raise ValueError('invalid checkpoint receipt')
    for name, expected in hashes.items():
        if (path/name).is_symlink() or sha(path/name) != expected:
            raise ValueError('incomplete or corrupted checkpoint: '+name)


def save(root, step, model, state, *, keep=3):
    import torch
    if keep < 2: raise ValueError('retain at least two recovery points')
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    final = root/f'checkpoint-{step:08d}'
    if final.exists():
        verify(final)
        raise ValueError('refusing to overwrite existing checkpoint')
    temp = Path(tempfile.mkdtemp(prefix=f'.saving-{step:08d}-',dir=root))
    model.save_pretrained(temp, safe_serialization=True)
    torch.save(state,temp/'state.pt')
    names = ('state.pt','adapter_config.json','adapter_model.safetensors')
    for name in names:
        with (temp/name).open('rb') as f: os.fsync(f.fileno())
    with (temp/'COMPLETE').open('w') as f:
        json.dump({n:sha(temp/n) for n in names},f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,final)
    verify(final)
    for old in complete(root)[:-keep]:
        if old.resolve().parent != root or old.is_symlink(): raise ValueError('unsafe prune target')
        verify(old)
        shutil.rmtree(old)
        print('PRUNED_CHECKPOINT='+str(old),flush=True)
    return final


def model_identity(base, adapter):
    from core import digest
    def files(path, patterns):
        result = {}
        for pattern in patterns:
            for f in sorted(Path(path).glob(pattern)):
                if f.is_symlink(): raise ValueError('unresolved model symlink')
                result[f.name] = sha(f)
        if not result: raise ValueError('missing model identity files')
        return result
    b = files(base, ('config.json','generation_config.json','*.safetensors','*.index.json'))
    a = files(adapter, ('adapter_config.json','adapter_model.safetensors'))
    tok = files(base, ('tokenizer*.json','special_tokens_map.json','added_tokens.json','chat_template.jinja'))
    return digest({'base':b,'adapter':a}), digest(tok)
