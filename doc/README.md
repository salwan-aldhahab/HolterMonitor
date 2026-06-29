# ECG Holter Monitor — Documentation

LaTeX source for the project's technical/developer manual.

## Layout

```
doc/
├── main.tex            # master document (compile this)
├── sections/           # one .tex per chapter, pulled in via \input
├── figures/            # drop device/case photos here (placeholders until then)
├── references.bib      # bibliography
└── README.md           # this file
```

## Building the PDF

You need a TeX distribution (TeX Live, MiKTeX, or MacTeX). The document uses only
standard packages (`tikz`, `listings`, `graphicx`, `hyperref`, …), so it also
compiles as-is on [Overleaf](https://overleaf.com) — just upload the `doc/` folder
and set `main.tex` as the main document.

### With latexmk (recommended)

```bash
cd doc
latexmk -pdf main.tex
```

### Manually

```bash
cd doc
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The output is `doc/main.pdf`.

## Adding real photos

The hardware section shows framed placeholder boxes for device photos. To replace
them, drop image files named as listed in [`figures/README.md`](figures/README.md)
into the `figures/` folder and rebuild — the placeholders are only drawn while the
image files are absent, so the document compiles either way.
