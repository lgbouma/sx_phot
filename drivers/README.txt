simplephot_slowest.py:
  Given RA/DEC, query IRSA to get all available images, then do simple aperture
  photometry and make a plot.

simplephot_slow.py:
  Given RA/DEC, query a mix of IRSA and AWS to get all available images, and do
  simple aperture photometry.  Makes a plot and a CSV table.  Better cacheing
  (e.g. including timestamps).
