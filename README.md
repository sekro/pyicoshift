# pyicoshift

This is a rewrite in python of the [icoshift](http://www.models.life.ku.dk/icoshift) algoritm by Francesco Savorani.
This implementation provides all the core features as icoshift for matlab version XXX.
## New, additional features:
 * New target mode: select the signal with the highest correlation with all input signals
 as target.
 * Auto interval picking: detected peaks get clustered based on distance and intervaled accordingly
 * Some utility functions for more convinient import of Bruker NMR data
 * Autophasing based on the "automics" algorithm
 * Experimental multi-threading support for the optimal shift finding phase
 * Experimental syntethic target generation from HMDB spectra
 
## Usage

Instance a new object of the class Icoshift, asign signals, choose intervals and target mode, call run()
; see scripts in test folder for some examples

