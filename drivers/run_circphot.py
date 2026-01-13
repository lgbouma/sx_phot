
## A star calib
#get_sx_spectrum(
#    ra_deg=0.8568686344700,
#    dec_deg=-46.830975607410
#)

## TOI-837
#get_sx_spectrum(
#    ra_deg=157.03727470138,
#    dec_deg=-64.50520903903,
#    star_id='TOI-837',
#    output_dir='test_results'
#)

from sx_phot.circphot import get_sx_spectrum

#get_sx_spectrum(
#        ra_deg=64.11714603424,
#        dec_deg=28.12648069800,
#        star_id='LkCa_4',
#        output_dir='test_results',
#        use_cutout=True
#)

get_sx_spectrum(
        ra_deg=114.36670927895,
        dec_deg=-66.75737858669,
        star_id='TIC_300651846',
        output_dir='test_results',
        use_cutout=True
)
#
#get_sx_spectrum(
#        ra_deg=108.97847939940,
#        dec_deg=-59.34149920022,
#        star_id='TIC_294328887',
#        output_dir='test_results',
#        use_cutout=True
#)

#get_sx_spectrum(
#        ra_deg=159.15796258208,
#        dec_deg=-64.79822926047,
#        star_id='TOI_6715',
#        output_dir='test_results',
#        use_cutout=True
#)
