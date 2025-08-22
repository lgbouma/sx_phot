------------------------------------------
Real drivers:

run_circphot.py:  Given RA/DEC, calls sx_phot.circphot to get the SPHEREx
  spectrum.  This is like circphot_zoom_simple below but enables wrapping.

------------------------------------------
Full pipeline scripts written in development:

simplephot_slowest.py:
  (Also, the simplest) Given RA/DEC, query IRSA to get all available images,
  then do simple aperture photometry and make a plot.

simplephot_slow.py:
  Given RA/DEC, query a mix of IRSA and AWS to get all available images, and do
  simple aperture photometry.  Makes a plot and a CSV table.  Better cacheing
  than simplephot_slowest.py (e.g. including timestamps).

circphot_slow.py:
  Given RA/DEC, query a mix of IRSA and AWS to get all available images, and do
  simple aperture photometry with either zodi or annulus subtraction.  Makes a
  plot and a CSV table.  Good cacheing both of fits images and of table (e.g.
  including timestamps).  Annulus seems preferable.

circphot_zoom_simple.py:
  Like circphot_slow.py, but way faster because it i) downloads IRSA cutouts by
  default rather than full images, and ii) multithreads over some other
  network-dependent steps.


