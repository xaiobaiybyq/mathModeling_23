from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parent

def main():
    validation=json.loads((ROOT/'results/validation.json').read_text('utf-8'))
    if validation['produced_samples'] != 100 or validation['structural_validation_errors']:
        raise SystemExit('Refusing to package incomplete or structurally invalid results')
    destination=ROOT/'question1_submission.zip'
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt','.md','.ps1']]
    files += list((ROOT/'results').rglob('*'))
    files += [ROOT/'models/model_manifest.json']
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files):
            if p.is_file():
                archive.write(p,'question1/'+p.relative_to(ROOT).as_posix())
    size=destination.stat().st_size
    if size > 50_000_000:
        raise SystemExit(f'Package exceeds 50 MB: {size} bytes')
    print(f'{destination.name}: {size} bytes ({size/1024**2:.2f} MiB)')
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert sum(n.endswith('.npz') for n in archive.namelist()) == 100
    print('Archive integrity and 100 feature files verified.')

if __name__=='__main__':
    main()
