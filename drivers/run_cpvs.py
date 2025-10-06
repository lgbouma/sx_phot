from sx_phot.circphot import get_sx_spectrum
import pandas as pd, numpy as np
from numpy import array as nparr
from glob import glob

df = pd.read_csv("data/20240304_CPV_lit_compilation_R16_S17_S18_B20_S21_Z19_G22_P23_B24_TIC8_obs_supplemented.csv")

ras, decs = nparr(df['tic8_ra']), nparr(df['tic8_dec'])
ticids = nparr(df['ticid_x'])

for ra, dec, ticid in zip(ras, decs, ticids):

    outpath = glob(f'cpv_results/*TIC_{ticid}*mjd*png')
    if len(outpath) == 0:
        get_sx_spectrum(
            ra_deg=ra,
            dec_deg=dec,
            star_id=f'TIC_{ticid}',
            output_dir='cpv_results',
            use_cutout=False
        )
    else:
        print(f'found {outpath}')

print('done 🎉')
