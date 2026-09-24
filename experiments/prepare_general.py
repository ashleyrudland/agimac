"""Bounded, pinned corpus acquisition and train-only BPE for the single-Mac pilot.

Uses public HTTP ranges for parquet and streaming gzip; no model inference/API.
Records source IDs, filters benchmark text overlap, and groups splits by source.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, HfApi
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from macoder.data import SPECIALS, CHAT_SPECIALS, file_hash

ROOT = Path('data/general-v1')
REV = '3ba9d605774198c5868892d7a8deda78031a781f'
CODE_REV = '3e6ab65f2864931e041f6a82db9b5a6ec2b71ab4'
TARGETS = {'fineweb': 32_000_000, 'cosmopedia': 24_000_000, 'python': 16_000_000, 'stories': 8_000_000}
# Overprovision characters so a trained tokenizer can supply the target tokens.
CHAR_LIMITS = {'fineweb': 180_000_000, 'cosmopedia': 150_000_000, 'python': 90_000_000, 'stories': 60_000_000}
EXPANDED = False
WORDS = re.compile(r'\w+')


def emit(**row):
    print(json.dumps(row), flush=True)


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'macoder-research/0.1'}), timeout=90)


def benchmark_filter():
    path = ROOT / 'benchmark-filter.json'
    if path.exists():
        return set(json.loads(path.read_text())['phrases'])
    reference = Path('recipes/reference/benchmark-filter.json')
    if reference.exists():
        shutil.copyfile(reference, path)
        return set(json.loads(path.read_text())['phrases'])
    urls = {
        'humaneval': 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz',
        'mbpp': 'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl',
    }
    phrases, sources = set(), {}
    for name, url in urls.items():
        blob = get(url).read()
        sources[name] = {'url': url, 'sha256': hashlib.sha256(blob).hexdigest()}
        text = gzip.decompress(blob).decode() if url.endswith('.gz') else blob.decode()
        for line in text.splitlines():
            row = json.loads(line)
            for key in ('prompt', 'canonical_solution', 'text', 'code', 'test', 'test_list'):
                value = row.get(key, '')
                if isinstance(value, list):
                    value = '\n'.join(value)
                words = WORDS.findall(value.lower())
                phrases.update(' '.join(words[i:i+13]) for i in range(len(words)-12))
    path.write_text(json.dumps({'sources': sources, 'phrases': sorted(phrases),
        'method': '13 consecutive lowercase words from prompts, solutions and tests; incomplete decontamination, not a guarantee'}))
    return phrases


def parquet_rows(subset):
    fs = HfFileSystem()
    filenames = fs.glob(f'datasets/HuggingFaceTB/smollm-corpus@{REV}/{subset}/train-*.parquet')
    for filename in sorted(filenames):
        with fs.open(filename, 'rb') as stream:
            table = pq.ParquetFile(stream)
            for group in range(table.num_row_groups):
                for row in table.read_row_group(group).to_pylist():
                    yield row, {'dataset': 'HuggingFaceTB/smollm-corpus', 'revision': REV,
                                'shard': filename.rsplit('/', 1)[1], 'row_group': group}


def rows(name):
    if name in ('fineweb', 'cosmopedia'):
        subset = 'fineweb-edu-dedup' if name == 'fineweb' else 'cosmopedia-v2'
        for row, provenance in parquet_rows(subset):
            if name == 'fineweb':
                identity = row['id']
                group = urlparse(row['metadata']['url']).hostname or identity
                record = {'id': identity, 'url': row['metadata']['url']}
            else:
                # Exact seed prompt family split; semantic near-duplicates may remain.
                group = hashlib.sha256(row['prompt'].encode()).hexdigest()
                record = {'prompt_sha256': group, 'seed_data': row.get('seed_data'), 'format': row.get('format')}
            yield row['text'], group, {**provenance, **record}
    elif name == 'python':
        filenames = ([f for f in HfApi().list_repo_files('codeparrot/codeparrot-clean-train', repo_type='dataset', revision=CODE_REV)
                      if f.endswith('.json.gz')] if EXPANDED else [f'file-{i:012d}.json.gz' for i in range(1,6)])
        for filename in sorted(filenames):
            url = f'https://huggingface.co/datasets/codeparrot/codeparrot-clean-train/resolve/{CODE_REV}/{filename}'
            with get(url) as response, gzip.GzipFile(fileobj=response) as gz:
                for line in gz:
                    row = json.loads(line)
                    if EXPANDED and row.get('license') not in {'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','cc0-1.0','unlicense'}:
                        continue
                    repo = row.get('repo_name') or row.get('repo')
                    if not repo:
                        raise ValueError(f'Missing repository key: {list(row)}')
                    yield row['content'], repo, {'dataset': 'codeparrot/codeparrot-clean-train',
                        'revision': CODE_REV, 'shard': filename, 'repo': repo,
                        'path': row.get('path'), 'license': row.get('license')}
    elif EXPANDED:
        revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
        url=f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/TinyStoriesV2-GPT4-train.txt'
        with get(url) as response:
            story=[]
            for raw in response:
                line=raw.decode('utf-8')
                if '<|endoftext|>' in line:
                    story.append(line.split('<|endoftext|>')[0]); text=''.join(story).strip();story=[]
                    if text:
                        yield text, hashlib.sha256(text.encode()).hexdigest(), {'dataset':'roneneldan/TinyStories','revision':revision,'url':url,'license':'cdla-sharing-1.0'}
                else:story.append(line)
    else:
        with Path('data/stories-source/train.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                yield row['text'], hashlib.sha256(row['text'].encode()).hexdigest(), {
                    'dataset': 'roneneldan/TinyStories', 'revision': 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
                    'source': 'existing 50000-document training prefix'}


def collect(name, blocked):
    source = ROOT / f'{name}.jsonl'
    meta_path = ROOT / f'{name}-source.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if file_hash(source) != meta['sha256']:
            raise ValueError(f'Changed staged source: {name}')
        return meta
    counts = {'train': 0, 'valid': 0}
    chars = {'train': 0, 'valid': 0}
    seen, rejected = set(), 0
    with source.open('w') as output:
        for text, group, provenance in rows(name):
            if not 150 <= len(text) <= 100_000:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            words = WORDS.findall(text.lower())
            if any(term in (str(provenance) + text[:2000]).lower() for term in ('humaneval', 'human_eval', 'evalplus', 'mbpp')) or any(
                ' '.join(words[i:i+13]) in blocked for i in range(len(words)-12)):
                rejected += 1
                continue
            seen.add(digest)
            split = 'valid' if int(hashlib.sha256((name+':'+group).encode()).hexdigest()[:8], 16) % 100 < 2 else 'train'
            output.write(json.dumps({'text': text, 'split': split, 'source': name,
                                     'group': group, 'sha256': digest, 'provenance': provenance}, ensure_ascii=False)+'\n')
            chars[split] += len(text)
            counts[split] += 1
            if sum(counts.values()) % 5000 == 0:
                emit(stage='collect', source=name, documents=counts, chars=chars)
            if chars['train'] >= CHAR_LIMITS[name] and chars['valid'] >= 300_000:
                break
    if not counts['valid']:
        raise ValueError(f'No validation documents: {name}')
    meta = {'source': name, 'documents': counts, 'characters': chars, 'overlap_rejections': rejected,
            'sha256': file_hash(source), 'bounded_prefix': True}
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    emit(stage='collected', **meta)
    return meta


def validation_hashes():
    hashes = set()
    for name in TARGETS:
        with (ROOT / f'{name}.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                if row['split'] == 'valid':
                    hashes.add(row['sha256'])
    return hashes


def tokenizer_documents(heldout):
    # Alternate sources; fit on at most 24M characters per source, training only.
    for name in TARGETS:
        chars = 0
        with (ROOT / f'{name}.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                if row['split'] == 'train' and row['sha256'] not in heldout:
                    yield row['text']
                    chars += len(row['text'])
                    if chars >= 24_000_000:
                        break


def pack():
    prepared = ROOT / 'prepared'
    prepared.mkdir(exist_ok=True)
    tok_path = prepared / 'tokenizer.json'
    # Use the original train-only BPE for comparable token IDs across machines.
    reference = Path('recipes/reference/tokenizer.json')
    if not tok_path.exists() and reference.exists():
        shutil.copyfile(reference, tok_path)
    heldout = validation_hashes()
    if not tok_path.exists():
        emit(stage='fit_tokenizer', vocabulary=16384)
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(tokenizer_documents(heldout), trainers.BpeTrainer(vocab_size=16384,
            special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        tok.save(str(tok_path))
    tok = Tokenizer.from_file(str(tok_path))
    # Append reserved message/tool tokens without changing any existing BPE IDs.
    tok.add_special_tokens(CHAT_SPECIALS)
    tok.save(str(tok_path))
    eos = tok.token_to_id(SPECIALS[0])
    actual, totals = {}, {'train': 0, 'valid': 0}
    seen = set()
    with (prepared / 'train.bin').open('wb') as train, (prepared / 'valid.bin').open('wb') as valid:
        for name, target in TARGETS.items():
            counts = {'train': 0, 'valid': 0}
            limits = {'train': target, 'valid': 100_000}
            with (ROOT / f'{name}.jsonl').open() as f:
                batch = []
                def write_batch(batch):
                    for row, encoded in zip(batch, tok.encode_batch([r['text'] for r in batch])):
                        split = row['split']
                        if row['sha256'] in seen or counts[split] >= limits[split] or (split == 'train' and row['sha256'] in heldout):
                            continue
                        seen.add(row['sha256'])
                        ids = (encoded.ids + [eos])[:limits[split] - counts[split]]
                        np.asarray(ids, dtype='<u4').tofile(train if split == 'train' else valid)
                        counts[split] += len(ids)
                for line in f:
                    row = json.loads(line)
                    if counts[row['split']] >= limits[row['split']]:
                        continue
                    batch.append(row)
                    if len(batch) == 128:
                        write_batch(batch)
                        batch = []
                    if all(counts[s] >= limits[s] for s in counts):
                        break
                if batch:
                    write_batch(batch)
            actual[name] = counts
            for split in totals:
                totals[split] += counts[split]
            emit(stage='packed_source', source=name, tokens=counts, target=target)
    if any(actual[n]['train'] < TARGETS[n]*(1.0 if EXPANDED else 0.8) for n in TARGETS):
        raise ValueError(f'Insufficient corpus: {actual}')
    sources = {name: json.loads((ROOT / f'{name}-source.json').read_text()) for name in TARGETS}
    manifest = {'vocab_size': tok.get_vocab_size(), 'chat_specials': CHAT_SPECIALS, 'tokenizer_sha256': file_hash(tok_path),
        'source_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        'sources': sources, 'tokens': totals, 'tokens_by_source': actual, 'dtype': '<u4', 'fim_rate': 0,
        'train_sha256': file_hash(prepared / 'train.bin'), 'valid_sha256': file_hash(prepared / 'valid.bin'),
        'split': '2% hash groups: website host / seed prompt / code repository / story hash',
        'decontamination': 'exact document dedup; HumanEval/MBPP 13-word overlap filter; NOT full near-duplicate decontamination',
        'benchmark_filter_sha256': file_hash(ROOT / 'benchmark-filter.json'),
        'expanded': EXPANDED, 'code_license_allowlist': ['mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','cc0-1.0','unlicense'] if EXPANDED else None}
    (prepared / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    emit(stage='ready', tokens=totals, path=str(prepared))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', default='data/general-v1')
    parser.add_argument('--tokens',type=int,default=80_000_000)
    parser.add_argument('--expanded',action='store_true')
    args=parser.parse_args()
    ROOT=Path(args.root);EXPANDED=args.expanded
    if args.tokens != 80_000_000 and not EXPANDED:parser.error('Larger corpus requires --expanded')
    if EXPANDED:
        TARGETS={k:int(args.tokens*f) for k,f in [('fineweb',.4),('cosmopedia',.3),('python',.2),('stories',.1)]}
        CHAR_LIMITS={k:int(v*7) for k,v in TARGETS.items()}
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'prepared/manifest.json').exists():
        emit(stage='already_prepared')
    else:
        blocked = benchmark_filter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda n: collect(n, blocked), TARGETS))
        pack()
