ADU_PER_ELECTRON = 8.6 # Detector gain
PIXEL_SIZE = 0.05 # Arcseconds
FORBIDDEN_KEYWORDS = "XTENSION BITPIX NAXIS NAXIS1 NAXIS2 NAXIS3 PCOUNT GCOUNT BSCALE BZERO EXTNAME".split() # Keywords which should not be copied between files
TRAP_T = 4 # Number of photoelectron traps
TRAP_P = 0.11 # Probability of trap catching a photoelectron
N_BIAS_FRAMES = 10_000 # Number of bias frames to use to measure bias