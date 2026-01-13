run_circphot.py:  Given RA/DEC, calls sx_phot.circphot to get the SPHEREx
  spectrum.  This is like scripts/circphot_zoom_simple below but enables wrapping.
get_tic_spherex_coords.py: Given a TIC ID, fetches Gaia DR2 proper motion and
  propagates RA/DEC to SPHEREx observation times, caching results under results/.
