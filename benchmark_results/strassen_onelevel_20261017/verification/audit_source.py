"""Audit existing instrument logs; instruction modifiers are not opcodes."""
import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
sass = (out / 'sass.txt').read_text()
# Match a SASS instruction line, excluding encoding lines and function names.
pattern = r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+\d+\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\b'
opcodes = Counter(op.split('.')[0] for op in re.findall(pattern, sass, re.M))
tensor_bases = ('HMMA', 'IMMA', 'DMMA', 'BMMA', 'MMA', 'WGMMA', 'HGMMA')
summary = {'ffma_instructions': opcodes['FFMA'],
           'tensor_instructions': sum(opcodes[k] for k in tensor_bases),
           'tensor_opcode_counts': {k: opcodes[k] for k in tensor_bases},
           'note': 'Counts instruction base opcodes, not modifiers; HFMA2.MMA is not a Tensor opcode.'}
assert summary['ffma_instructions'] > 0 and summary['tensor_instructions'] == 0
for name in ('profile_small', 'tensor_cublas'):
    content = (out / f'{name}.csv').read_text()
    rows = list(csv.DictReader(io.StringIO(content[content.index('"ID"'):])))
    metrics = {r.get('Metric Name'): r.get('Metric Value') for r in rows if r.get('Metric Name')}
    if name == 'profile_small':
        wanted = ('Registers Per Thread', 'Theoretical Occupancy', 'Achieved Occupancy', 'Waves Per SM')
        summary['small_profile'] = {k: metrics[k] for k in wanted}
    else:
        value = metrics['sm__inst_executed_pipe_tensor.sum']
        assert float(value.replace(',', '')) == 0
        summary['cublas_tensor_instructions'] = value
for name in ('memcheck_mask', 'memcheck_huge', 'racecheck'):
    content = (out / f'{name}.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in content or 'RACECHECK SUMMARY: 0 hazards' in content
    checks = list(csv.DictReader(open(out / f'{name}.correctness.csv')))
    assert checks and all(r['pass'] == '1' for r in checks)
    summary[f'{name}_fp64_checks'] = len(checks)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
