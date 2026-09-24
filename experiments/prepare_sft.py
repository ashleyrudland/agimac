"""Download a pinned public conversation subset; prepare complete masked examples."""
from collections import Counter
import hashlib,json,random,re,shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer
from macoder.conversation import render_messages,execute_calculator,TOOL_SYSTEM,FORMAT
from macoder.data import file_hash

REV='f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'
ROOT=Path('data/sft-v1');ROOT.mkdir(parents=True,exist_ok=True)
BASE=Path('runs/general-v1/step-048829')
TOK=Tokenizer.from_file(str(BASE/'tokenizer.json'))
BLOCKED=set(json.loads(Path('data/general-v1/benchmark-filter.json').read_text())['phrases'])
WORDS=re.compile(r'\w+')
BUCKETS=(256,512,1024)


def emit(**x):print(json.dumps(x),flush=True)

def key(messages):
    return hashlib.sha256(json.dumps([m['content'] for m in messages if m['role']=='user'],ensure_ascii=False).encode()).hexdigest()

def source_rows(split):
    fs=HfFileSystem()
    for path in sorted(fs.glob(f'datasets/HuggingFaceTB/smol-smoltalk@{REV}/data/{split}-*.parquet')):
        with fs.open(path,'rb') as stream:
            pf=pq.ParquetFile(stream)
            for g in range(pf.num_row_groups):
                yield from pf.read_row_group(g).to_pylist()

def suitable(messages):
    if not messages or messages[-1]['role']!='assistant':return None
    if any(not isinstance(m.get('content'),str) for m in messages):return None
    body=' '.join(m['content'] for m in messages)
    words=WORDS.findall(body.lower())
    if any(' '.join(words[i:i+13]) in BLOCKED for i in range(len(words)-12)):return None
    try: ids,mask=render_messages(TOK,messages)
    except (ValueError,KeyError):return None
    if len(ids)>1025 or sum(mask)<8:return None
    return ids,mask


def main():
    if (ROOT/'manifest.json').exists():emit(stage='already_ready');return
    outputs={s:(ROOT/f'{s}.jsonl').open('w') for s in ('train','valid','test')}
    counts=Counter();sources=Counter();seen=set();heldout=set()
    def save(split,messages,source,encoded=None):
        k=key(messages)
        if k in seen:return False
        result=encoded or suitable(messages)
        if result is None:return False
        seen.add(k)
        if split!='train':heldout.add(k)
        outputs[split].write(json.dumps({'messages':messages,'source':source,'prompt_hash':k})+'\n')
        counts[split]+=1;sources[f'{split}/{source}']+=1
        return True
    # Freeze two disjoint held-out subsets before examining training rows.
    for row in source_rows('test'):
        split='valid' if counts['valid']<1000 else 'test'
        save(split,row['messages'],row.get('source','smoltalk'))
        if counts['test']>=500:break
    for row in source_rows('train'):
        if key(row['messages']) in heldout:continue
        if save('train',row['messages'],row.get('source','smoltalk')) and counts['train']%10000==0:
            emit(stage='collect_sft',counts=dict(counts))
        if counts['train']>=150000:break
    rng=random.Random(9271)
    for _ in range(18000):
        name=rng.choice(['add','subtract','multiply','divide']);a=rng.randint(-99,99);b=rng.randint(1,99)
        # Integer divisions keep targets short and exact.
        if name=='divide':a*=b
        call={'name':name,'arguments':{'a':a,'b':b}}
        value=execute_calculator(call)
        prompt=f'Use the calculator to {name} {a} and {b}.'
        messages=[{'role':'system','content':TOOL_SYSTEM},{'role':'user','content':prompt},
                  {'role':'assistant','tool_call':call},{'role':'tool','content':json.dumps(value,separators=(',',':'))},
                  {'role':'assistant','content':str(value['result'])}]
        bucket=int(key(messages)[:8],16)%100
        split='test' if bucket<2 else ('valid' if bucket<4 else 'train')
        save(split,messages,'verified_calculator',render_messages(TOK,messages))
    for f in outputs.values():f.close()
    shutil.copyfile(BASE/'tokenizer.json',ROOT/'tokenizer.json')
    packed={}
    for split in outputs:
        groups={b:[] for b in BUCKETS}
        for line in (ROOT/f'{split}.jsonl').read_text().splitlines():
            row=json.loads(line);ids,mask=render_messages(TOK,row['messages'])
            bucket=next(b for b in BUCKETS if len(ids)<=b+1)
            groups[bucket].append((ids,mask))
        packed[split]={}
        for bucket,items in groups.items():
            ids=np.zeros((len(items),bucket+1),dtype=np.uint32)
            mask=np.zeros_like(ids,dtype=np.uint8)
            actual=supervised=0
            for i,(t,m) in enumerate(items):
                ids[i,:len(t)]=t;mask[i,:len(m)]=m
                actual+=len(t)-1;supervised+=sum(m[1:])
            np.save(ROOT/f'{split}-{bucket}-ids.npy',ids);np.save(ROOT/f'{split}-{bucket}-mask.npy',mask)
            packed[split][bucket]={'examples':len(items),'actual_input_tokens':actual,'assistant_targets':supervised}
        emit(stage='packed',split=split,buckets=packed[split])
    hashes={p.name:file_hash(p) for p in ROOT.glob('*.npy')}
    manifest={'dataset':'HuggingFaceTB/smol-smoltalk','revision':REV,'chat_format':FORMAT,
              'tokenizer_sha256':file_hash(ROOT/'tokenizer.json'),'counts':dict(counts),'sources':dict(sources),
              'buckets':packed,'array_hashes':hashes,'jsonl_hashes':{s:file_hash(ROOT/f'{s}.jsonl') for s in outputs},
              'selection':'First 150K unique prompt groups with complete conversations <=1025 tokens; no response truncation',
              'split':'Official test prefix split into 1000 development and 500 final examples; exact user-prompt groups excluded from train. Calculator split by prompt hash.',
              'decontamination':'Same 13-word HumanEval/MBPP exclusion filter for SmolTalk; not exhaustive semantic decontamination',
              'benchmark_filter_sha256':file_hash(Path('data/general-v1/benchmark-filter.json'))}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');emit(stage='sft_ready',counts=dict(counts))

if __name__=='__main__':main()
