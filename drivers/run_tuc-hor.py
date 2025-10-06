from sx_phot.circphot import get_sx_spectrum
from astropy.table import Table
import pandas as pd, numpy as np
from numpy import array as nparr

t = Table.read("data/apjacb055t1_mrt.txt", format='mrt')
df = t.to_pandas()

df['bp_rp'] = df['BPmag'] - df['RPmag']
sel = df['bp_rp'] > 2.0
df = df[sel]

ras, decs = nparr(df['RAdeg']), nparr(df['DEdeg'])
dr3_sourceids = nparr(df['DR3'])

for ra, dec, dr3_sourceid in zip(ras, decs, dr3_sourceids):

    outpath = glob(f'tuc-hor_results/*DR3_{dr3_sourceids}*mjd*png')
    if len(outpath) == 0:
        get_sx_spectrum(
            ra_deg=ra,
            dec_deg=dec,
            star_id=f'DR3_{dr3_sourceid}',
            output_dir='tuc-hor_results',
            use_cutout=False
        )
    else:
        print(f'found {outpath}')

print('done 🎉')
