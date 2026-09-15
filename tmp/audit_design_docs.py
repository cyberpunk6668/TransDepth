from pathlib import Path
import ast
import hashlib
import json
import re
import tomllib

root = Path(__file__).resolve().parents[1]
docs = [root / 'README.md', *sorted((root / 'tech_documents/implementation').glob('*.md'))]
errors, entries, links = [], [], 0
for path in docs:
    s = path.read_text(encoding='utf-8')
    if '\x00' in s or '\ufffd' in s:
        errors.append(f'{path.name}: invalid text character')
    active = None
    start = 0
    chunks = []
    for lineno, line in enumerate(s.splitlines(), 1):
        m = re.match(r'^(`{3,}|~{3,})(\w*)\s*$', line)
        if not m:
            if active:
                chunks.append(line)
            continue
        if active is None:
            active = (m[1][0], len(m[1]), m[2])
            start, chunks = lineno, []
        elif m[1][0] == active[0] and len(m[1]) >= active[1]:
            code = '\n'.join(chunks)
            try:
                if active[2] == 'python':
                    ast.parse(code)
                if active[2] == 'toml':
                    tomllib.loads(code)
                if active[2] == 'json':
                    json.loads(code)
            except Exception as exc:
                errors.append(f'{path.name}:{start}: {active[2]}: {exc}')
            active = None
    if active:
        errors.append(f'{path.name}: unclosed code fence at {start}')
    prose = re.sub(r'^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$', '', s, flags=re.M | re.S)
    prose = re.sub(r'`[^`\n]+`', '', prose)
    for m in re.finditer(r'\[[^\]\n]*\]\((<[^>]+>|[^)]+)\)', prose):
        target = m[1].strip('<>')
        if '://' in target or target.startswith('#'):
            continue
        # Local file references only; ignore anchor portion.
        target = target.split('#')[0]
        if target and not (path.parent / target).exists():
            errors.append(f'{path.name}: missing link target {target}')
        links += 1
    entries.append({'file': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                    'lines': len(s.splitlines()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
expected = [f'{i:02d}_' for i in range(8)]
for prefix in expected:
    if not any(p.name.startswith(prefix) for p in docs):
        errors.append(f'missing document {prefix}')
report = {'document_count': len(entries), 'local_links_checked': links, 'files': entries,
          'errors': errors, 'status': 'pass' if not errors else 'fail',
          'scope': 'document structure, links, Python syntax, TOML and JSON syntax only; not execution'}
(root / 'tmp/design_docs_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'files'}, ensure_ascii=False, indent=2))
