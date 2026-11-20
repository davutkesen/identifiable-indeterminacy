MATLAB code for the four figures of

Identifiable Indeterminacy: A Per-Feature Neutrosophic Metric for
Nearest-Neighbour Classification
=================================================================

  make_figures.m     writes Fig1.eps ... Fig4.eps into figures/
  figprolog.ps       the two fonts and the drawing procedures of the
                     figures; make_figures copies it into every file.  Its
                     head states the licence of the fonts.
  data/              fig3_scores.csv and fig4_tasks.csv, the numbers of
                     Figures 3 and 4.  Figures 1 and 2 need no data.
  reference.txt      size and Adler-32 checksum of the four EPS files the
                     article is typeset from

RUNNING
  In MATLAB, from this folder (no toolbox is needed):

      make_figures
      make_figures(outdir)        writes into another folder

  The function does not call MATLAB's plotting functions.  It writes the
  PostScript of each figure itself: every coordinate is a whole number of
  thousandths of a point, computed in integer arithmetic, and all text is
  measured and placed by the PostScript interpreter with the fonts of
  figprolog.ps.  Nothing in the files therefore depends on the release, the
  operating system, the screen or the installed fonts.

  After writing, the function compares each file with reference.txt and
  prints one line per figure:

      Fig1.eps is identical to the figure of the article (... bytes)

  The files of the article are

      Fig1.eps    140168 bytes   MD5 c8d22e71c71b5333caeb74e0adf6d009
      Fig2.eps     94398 bytes   MD5 cac87c693de4297f76ae1f6dc80fe269
      Fig3.eps     61091 bytes   MD5 093fca5d22fa0033e3627849443f3fec
      Fig4.eps     69321 bytes   MD5 0b9491ed068f1d18990a1ce7e7c4c7b4

  and they are the files Fig1.eps ... Fig4.eps of the LaTeX source of the
  article.

USING THE FILES IN THE ARTICLE
  Copy Fig1.eps ... Fig4.eps into the folder of manuscript.tex.  That folder
  also holds Fig1.pdf ... Fig4.pdf, which were converted from the EPS files
  with Ghostscript,

      gs -q -dNOPAUSE -dBATCH -dSAFER -dEPSCrop -sDEVICE=pdfwrite
         -dAutoRotatePages=/None -sOutputFile=Fig1.pdf Fig1.eps

  and pdflatex reads those.  To have pdflatex read the EPS files instead,
  delete the four PDF files: it then converts each EPS file on the fly
  (package epstopdf, part of every TeX distribution).  Both ways give the
  same pages, pixel for pixel.  latex followed by dvips reads the EPS files
  directly and sets the same drawing on the grid of its output device, a
  shift of a fraction of a pixel.

THE DATA
  The two files were written from the stored results in code/experiments/,
  with every score and coordinate rounded to six decimals and stored as a
  whole number of millionths.

  fig3_scores.csv   132 rows, from weights_k.json (dipeptide composition)
      panel             1 direct construction, 2 proposed construction
      k                 the selected number of neighbours
      w_indeterminacy   weight of the indeterminacy block, in tenths
      w_first           weight of T (panel 1) or M0 (panel 2), in tenths
      w_second          weight of F (panel 1) or M1 (panel 2), in tenths
      score             cross-validated macro F1 at the selected k and
                        class weighting, in millionths

  fig4_tasks.csv    319 rows, from regime.json (panel 1), regime_crossfit.json
                    and regime_pmlb.json (panel 2)
      panel             1 fixed configuration, 2 both rules tuned and
                        coordinates cross-fitted
      source            1 to 4 for the sources A to D
      x                 macro F1 of plain k-NN minus that of the majority
                        classifier, in millionths
      y                 macro F1 gained by the transformation, in millionths

HOW IT WAS TESTED
  make_figures.m is a line-by-line port of code/experiments/make_figures.py,
  which wrote the figures of the article.  The port was run
  under GNU Octave 8.4 and its four files were compared byte for byte with
  those of the Python script: they are identical.  It was not run under
  MATLAB itself.  It uses only operations that are exact in double
  precision (sums, products and rounded quotients of whole numbers far
  below 2^53) and text functions that MATLAB and Octave define alike, and
  the line printed for each figure shows on any machine whether the result
  is the figure of the article.
