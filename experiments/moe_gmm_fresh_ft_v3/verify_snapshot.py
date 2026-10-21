"""Verify the sealed execution snapshot without importing training code."""
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    lock=json.loads((root/'code/source_manifest.json').read_text(encoding='utf-8'))
    for name,expected in lock.items():
        p=root/'code'/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('sealed file changed: '+name)
    gpu=json.loads((root/'evidence/validation/gpu.json').read_text(encoding='utf-8'))
    review=json.loads((root/'evidence/review_package_validation.json').read_text(encoding='utf-8'))
    assert gpu['passed'] and gpu['source_manifest']==lock
    assert review['cpu_tests_passed']==12 and review['gpu_validation_passed']
    assert review['gpu']['source_manifest']==lock
    p=root/'PUBLICATION_MANIFEST.json'
    if p.exists():
        for name,expected in json.loads(p.read_text(encoding='utf-8'))['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('publication file changed: '+name)
    print(f'PASS: {len(lock)} sealed source/config files; matching GPU/review source locks')

if __name__=='__main__':main()
