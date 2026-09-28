"""Map official CBIS-DDSM mass metadata to local crop files, without guessing.

Exact paths are tried first, then unique SeriesInstanceUID matches. Ambiguous
series require an explicit override. No patient-path substring matching is used.
"""
from pathlib import Path, PurePosixPath
import re
import pandas as pd
from project import canonical_patient


def build_manifest(train_csv, test_csv, image_root, destination, overrides=None):
    image_root, destination = Path(image_root).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Manifest already exists; preserve it or choose a new output name')
    if not image_root.is_dir():
        raise FileNotFoundError(f'Image root is not a folder: {image_root}')
    lookup = {}
    if overrides is not None:
        override_path = Path(overrides).resolve()
        override_df = pd.read_csv(override_path, dtype=str, keep_default_na=False)
        if override_df.source_path.duplicated().any():
            raise ValueError('Duplicate source_path in overrides')
        for r in override_df.itertuples(index=False):
            p = Path(r.local_path)
            lookup[r.source_path.strip().replace('\\', '/')] = (
                p if p.is_absolute() else override_path.parent / p).resolve()
    rows = []
    required = {'patient_id', 'pathology', 'cropped image file path'}
    for split, path in [('train', train_csv), ('test', test_csv)]:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = df.columns.str.strip()
        if not required <= set(df.columns):
            raise ValueError(f'{path} is missing official mass metadata columns')
        for _, r in df.iterrows():
            source = r['cropped image file path'].strip().replace('\\', '/')
            if not source:
                raise ValueError('Blank cropped image file path; inspect source CSV')
            row = {'patient_id': canonical_patient(r['patient_id']), 'pathology': r['pathology'],
                   'official_split': split, 'source_path': source,
                   'breast_density': r.get('breast_density', r.get('breast density', '')),
                   'image_view': r.get('image view', ''),
                   'lesion_id': r.get('abnormality id', '')}
            candidate = lookup.get(source, image_root / Path(source))
            row['image_path'] = str(candidate.resolve()) if candidate.is_file() else ''
            rows.append(row)
    # Header indexing is required only for files whose original hierarchy changed.
    unresolved = [r for r in rows if not r['image_path']]
    series_index, unreadable = {}, []
    if unresolved:
        import pydicom
        paths = sorted(p for p in image_root.rglob('*') if p.suffix.lower() in {'.dcm', '.dicom'})
        print(f'Indexing {len(paths)} DICOM headers; this may take a few minutes.')
        for p in paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True,
                                    specific_tags=['SeriesInstanceUID'])
                uid = str(getattr(ds, 'SeriesInstanceUID', ''))
                if uid:
                    series_index.setdefault(uid, []).append(str(p.resolve()))
            except Exception as exc:
                unreadable.append({'path': str(p), 'error': str(exc)})
    failures = []
    for row in rows:
        if row['image_path']:
            continue
        parts = PurePosixPath(row['source_path']).parts
        uid = parts[-2] if len(parts) >= 2 else ''
        candidates = series_index.get(uid, []) if re.fullmatch(r'[0-9.]+', uid) else []
        if len(candidates) == 1:
            row['image_path'] = candidates[0]
        else:
            failures.append({'source_path': row['source_path'], 'local_path': '',
                'reason': 'ambiguous series' if candidates else 'no exact path or unique series match',
                'candidates': ' | '.join(candidates)})
    pd.DataFrame(unreadable, columns=['path', 'error']).to_csv(
        destination.parent / 'unreadable_dicom_headers.csv', index=False)
    if failures:
        failure_path = destination.parent / 'path_mapping_needed.csv'
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        raise ValueError(f'{len(failures)} mappings unresolved. Fill local_path in {failure_path}, '
                         'then pass it as overrides. Choose the lesion IMAGE, not its ROI mask.')
    result = pd.DataFrame(rows)
    result.to_csv(destination, index=False)
    print(f'Saved {len(result)} manifest rows to {destination}')
    return result

