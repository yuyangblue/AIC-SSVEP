#!/usr/bin/env python3
'''校验 papers 目录下所有 PDF：头魔数、尾部 EOF、页数近似、是否含标题关键词。'''
import os, re, glob

papers = os.path.expanduser('~/workspace/aic/papers')
pdfs = sorted(glob.glob(os.path.join(papers, '*.pdf')))
print(f'共 {len(pdfs)} 个 PDF')
print('-' * 78)
all_ok = True
for p in pdfs:
    raw = open(p, 'rb').read()
    head = raw[:5] == b'%PDF-'
    tail = raw[-2048:].rstrip()
    eof = tail.endswith(b'%%EOF') or b'%%EOF' in raw[-4096:]
    npage = len(re.findall(rb'/Type\s*/Page(?!s)', raw))
    size_mb = len(raw) / 1e6
    ok = head and eof and npage >= 3 and size_mb > 0.15
    all_ok &= ok
    flag = 'OK ' if ok else 'BAD'
    print(f'[{flag}] {os.path.basename(p):48s} {size_mb:5.2f}MB  pages~{npage:3d}  head={head} eof={eof}')
print('-' * 78)
print('ALL OK' if all_ok else 'SOME FILES NEED ATTENTION')
