"""Parse GEO series matrices -> per-sample table (GSM, title, source, characteristics) for mouse sets."""
import gzip, glob, re, pandas as pd
rows = []
for f in sorted(glob.glob('./geo/*_series_matrix.txt.gz')):
    gse = re.match(r'.*/(GSE\d+)', f).group(1)
    acc = None; chars = []; src = None; title = None; org = None
    for line in gzip.open(f, 'rt', errors='ignore'):
        if line.startswith('!Sample_geo_accession'): acc = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'): chars.append([x.strip('"') for x in line.rstrip('\n').split('\t')[1:]])
        elif line.startswith('!Sample_source_name_ch1'): src = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
        elif line.startswith('!Sample_title'): title = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
        elif line.startswith('!Sample_organism_ch1'): org = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
        elif line.startswith('!series_matrix_table_begin'): break
    for i, g in enumerate(acc):
        d = {'GSE': gse, 'GSM': g, 'title': title[i] if title else '', 'source': src[i] if src else '', 'organism': org[i] if org else ''}
        for c in chars:
            if i < len(c) and ':' in c[i]:
                k, v = c[i].split(':', 1); d[k.strip().lower()] = v.strip()
        rows.append(d)
m = pd.DataFrame(rows).drop_duplicates('GSM')
m.to_csv('./geo/mouse_geo_meta.csv', index=False)
print(m.groupby('GSE').size())
for g, d in m.groupby('GSE'):
    agecols = [c for c in d.columns if 'age' in c and d[c].notna().any()]
    print(g, agecols, d[agecols].drop_duplicates().head(4).values.tolist() if agecols else d[['title']].head(3).values.tolist())
