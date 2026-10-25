"""One-to-one human-mouse orthologs (NCBI gene_orthologs) for protein-coding genes, with promoter windows
(TSS -1500/+500, strand-aware) from Ensembl GRCm38 (mm10) and GRCh37 (hg19) GTFs."""
import gzip, re, pandas as pd, numpy as np
A = './ann/'
o = pd.read_csv(A + 'gene_orthologs.gz', sep='\t')
o = o[(o['#tax_id'] == 9606) & (o.Other_tax_id == 10090)][['GeneID', 'Other_GeneID']]
o.columns = ['hs_id', 'mm_id']
# one-to-one only
o = o[~o.hs_id.duplicated(keep=False) & ~o.mm_id.duplicated(keep=False)]
gi = {}
for sp, f in [('hs', 'Homo_sapiens'), ('mm', 'Mus_musculus')]:
    g = pd.read_csv(A + f + '.gene_info.gz', sep='\t', usecols=['GeneID', 'Symbol', 'type_of_gene'])
    gi[sp] = g[g.type_of_gene == 'protein-coding'].set_index('GeneID')['Symbol']
o['hs'] = o.hs_id.map(gi['hs']); o['mm'] = o.mm_id.map(gi['mm'])
o = o.dropna()


def tss(gtf):
    rows = []
    for line in gzip.open(gtf, 'rt'):
        if line.startswith('#'): continue
        f = line.split('\t', 8)
        if f[2] != 'gene' or 'gene_biotype "protein_coding"' not in f[8]: continue
        name = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
        chrom = f[0] if f[0].startswith('chr') else 'chr' + f[0]
        s, e, strand = int(f[3]), int(f[4]), f[6]
        t = s if strand == '+' else e
        lo, hi = (t - 1500, t + 500) if strand == '+' else (t - 500, t + 1500)
        rows.append((name, chrom, lo, hi, strand, s, e))
    d = pd.DataFrame(rows, columns=['gene', 'chr', 'lo', 'hi', 'strand', 'gstart', 'gend'])
    return d.drop_duplicates('gene', keep=False)       # drop symbols with several loci


tm, th = tss(A + 'mm10.gtf.gz'), tss(A + 'hg19.gtf.gz')
o = o[o.mm.isin(tm.gene) & o.hs.isin(th.gene)]
o.to_csv(A + 'ortho_1to1.csv', index=False)
tm[tm.gene.isin(o.mm)].to_csv(A + 'mm10_promoters.csv', index=False)
th[th.gene.isin(o.hs)].to_csv(A + 'hg19_promoters.csv', index=False)
print('one-to-one protein-coding orthologs with unique TSS in both genomes:', len(o))
