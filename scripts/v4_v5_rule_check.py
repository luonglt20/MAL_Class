"""Compare v4/v5 rule statuses against CAPE behavioral signatures on raw reports (JSON only)."""
import json, sys, pathlib, collections, py7zr
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.evidence import EvidenceEngine
archive_path, out, n = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
with py7zr.SevenZipFile(archive_path, password="infected") as a:
    names = [i.filename for i in a.list() if i.filename.endswith('.json')][:n]
todo = [x for x in names if not (out / x).exists()]
if todo:
    with py7zr.SevenZipFile(archive_path, password="infected") as a:
        a.extract(path=out, targets=todo)
PAIRS = {'T1547.001': {'persistence_autorun'}, 'T1055': {'injection_rwx', 'injection_createremotethread',
         'injection_write_memory', 'injection_runpe', 'process_interest', 'injection_ntsetcontextthread'},
         'T1543.003': {'creates_service', 'persistence_service'}, 'T1486': {'ransomware_files', 'ransomware_extensions'}}
eng = EvidenceEngine(); tab = collections.defaultdict(collections.Counter); seen = collections.Counter()
for name in names:
    raw = json.loads((out / name).read_text())
    for ver, v5 in (('v4', False), ('v5', True)):
        r = CapeReportParser(file_handle_resolution=True, registry_value_resolution=v5).parse(raw)
        m = eng.evaluate(r.events)
        sigs = set(CapeReportParser(file_handle_resolution=True, registry_value_resolution=True).parse(raw).metadata['cape_signatures']) if ver == 'v4' else set(r.metadata['cape_signatures'])
        if ver == 'v5': seen.update(sigs)
        for t, ref in PAIRS.items():
            tab[(ver, t)][(m[t].status, bool(sigs & ref))] += 1
print(json.dumps({'reports': len(names)}))
for (ver, t), c in sorted(tab.items(), key=lambda x: (x[0][1], x[0][0])):
    conf_sig = c[('confirmed', True)]; conf_nosig = c[('confirmed', False)]; sig = sum(v for (s, h), v in c.items() if h)
    print(t, ver, 'sig_positive', sig, 'confirmed&sig', conf_sig, 'confirmed&no_sig', conf_nosig,
          'suspected&sig', c[('suspected', True)])
print('top signatures', seen.most_common(25))
