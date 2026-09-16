"""Build an explicit allowlisted source archive and inspectable public staging tree.

No credentials, live Core source/backups, installed copies or local release notes.
This creates build artifacts only. Run from the project root after tests/wheel build.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.2.0rc2'
FILES = {
    'README.md': 'PUBLIC_README.md',
    'PUBLIC_README.md': 'PUBLIC_README.md',
    'LICENSE_NOTICE.md': 'LICENSE_NOTICE.md',
    'AUTHORS.md': 'AUTHORS.md',
    'pyproject.toml': 'pyproject.toml',
    'engine.py': 'engine.py',
    'test_engine.py': 'test_engine.py',
    'test_integration.py': 'test_integration.py',
    'test_cli.py': 'test_cli.py',
    'marketing/reddit-launch.png': 'marketing/reddit-launch.png',
    'marketing/REDDIT_POST_EN.md': 'marketing/REDDIT_POST_EN.md',
    'marketing/LAUNCH_PLAN.md': 'marketing/LAUNCH_PLAN.md',
    'marketing/IMAGE_PROMPT_FULL.md': 'marketing/IMAGE_PROMPT_FULL.md',
}
for name in ('__init__.py', '__main__.py', 'cli.py', 'client.py', 'engine.py',
             'service.py', 'http.py', 'openapi.json', 'py.typed'):
    FILES['specter_decision/' + name] = 'specter_decision/' + name


def main():
    stage = ROOT / 'release' / 'public-source'
    stage.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for public, local in FILES.items():
        data = (ROOT / local).read_bytes()
        if not public.endswith('.png'):
            content = data.decode('utf-8')
            if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----', content):
                raise SystemExit('SECRET_PATTERN_FOUND: publication aborted; matched value suppressed')
        payloads[public] = data
    # Never delete unexpected files. Stop if staging was changed by another actor.
    unexpected = [p for p in stage.rglob('*') if p.is_file()
                  and '.git' not in p.relative_to(stage).parts
                  and p.relative_to(stage).as_posix() not in FILES]
    if unexpected:
        raise SystemExit('UNEXPECTED_STAGING_FILES')
    for public, data in payloads.items():
        target = stage / public
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    archive = ROOT / 'dist' / ('specter-decision-engine-' + VERSION + '-source.zip')
    with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
        for public, data in payloads.items():
            output.writestr('specter-decision-engine-' + VERSION + '/' + public, data)
    wheel = ROOT / 'dist' / ('specter_decision_engine-' + VERSION + '-py3-none-any.whl')
    with ZipFile(wheel) as package:
        if not all(n.startswith(('specter_decision/', 'specter_decision_engine-' + VERSION + '.dist-info/')) for n in package.namelist()):
            raise SystemExit('UNEXPECTED_WHEEL_CONTENT')
        for public, data in payloads.items():
            if public.startswith('specter_decision/') and package.read(public) != data:
                raise SystemExit('WHEEL_SOURCE_MISMATCH')
    sums = []
    for file in (wheel, archive):
        sums.append(hashlib.sha256(file.read_bytes()).hexdigest() + '  ' + file.name)
    (ROOT / 'dist' / 'SHA256SUMS').write_text('\n'.join(sums) + '\n', encoding='utf-8')
    print(json.dumps({'version': VERSION, 'public_files': len(payloads),
                      'secret_pattern_scan': 'PASS', 'wheel_source_match': True,
                      'staging': str(stage), 'sha256sums': sums}))


if __name__ == '__main__': main()
