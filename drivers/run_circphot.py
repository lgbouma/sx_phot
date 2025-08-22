from sx_phot.circphot import get_sx_spectrum

# # A star calib
# get_sx_spectrum(
#     ra_deg=0.8568686344700,
#     dec_deg=-46.830975607410
# )

# TOI-837
get_sx_spectrum(
    ra_deg=157.03727470138,
    dec_deg=-64.50520903903,
    star_id='TOI-837',
    output_dir='test_results'
)
