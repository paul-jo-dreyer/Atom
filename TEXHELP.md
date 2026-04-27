# LaTeX workflow

Reference for compiling the analysis documents in this repo (`AtomSim/dynamics/<vehicle>/analysis/diff_drive.tex` and friends). The main document is split into a thin shell + `\input`-ed section files; this guide covers the compile commands and how to set up watch-on-save so you can iterate without thinking about it.

## One-time setup

Install a TeX distribution and the `latexmk` build wrapper:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended latexmk
```

`texlive-latex-extra` brings in `pdflatex` plus the packages we use (`amsmath`, `amssymb`, `mathtools`, `hyperref`, `geometry`, `xcolor`). `latexmk` is the convenience wrapper that handles re-runs for you. ~150 MB total.

If you only have `pdflatex` (no `latexmk`), see the manual recipe at the bottom.

## Build once

From the directory containing the main `.tex` file:

```bash
cd AtomSim/dynamics/diff_drive/analysis
latexmk -pdf diff_drive.tex
```

`latexmk` runs `pdflatex` as many times as needed for the table of contents and cross-references to settle, then stops. Output is `diff_drive.pdf` in the same directory.

## Build on save (watch mode)

```bash
latexmk -pdf -pvc diff_drive.tex
```

`-pvc` (**p**review **c**ontinuously) watches the source files — including all `\input`-ed section files — and rebuilds whenever any of them change. Leave it running in a terminal while you edit; the PDF updates in place. `Ctrl-C` to stop.

Pair with a PDF viewer that auto-reloads on file change (`zathura`, `evince`, Okular, or VSCode's built-in PDF viewer) and you have an effectively live preview.

## Clean up build artifacts

```bash
latexmk -c    # remove .aux, .log, .toc, .out (keep PDF)
latexmk -C    # also remove the PDF
```

These artifacts are gitignored at the repo root (`*.aux`, `*.log`, `*.out`, `*.toc`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.bbl`, `*.blg`), so leaving them in place doesn't pollute commits — but they do clutter `ls`.

## VSCode integration

Install the **LaTeX Workshop** extension (`james-yu.latex-workshop`). It autodetects `latexmk` and gives you:

- Build-on-save (toggle: command palette → *LaTeX Workshop: Toggle build on save*).
- An in-editor PDF preview pane (right-click in the `.tex` editor → *Open in VSCode tab*).
- Forward/inverse search via SyncTeX: <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>J</kbd> from a source line jumps to the matching spot in the PDF; <kbd>Ctrl</kbd>+click in the PDF jumps back to the source line.

**Important**: builds always run against the file with `\documentclass`, which is `diff_drive.tex`. If you're focused on a section file under `sections/` and trigger a build, LaTeX Workshop is smart enough to find the parent — but only if you've opened the project from a folder that contains `diff_drive.tex` somewhere underneath. Open VSCode at the repo root (`Atom/`) for everything to resolve correctly.

If you want to opt this project into LaTeX Workshop's `latexmk` recipe explicitly, add to `.vscode/settings.json`:

```json
"latex-workshop.latex.recipe.default": "latexmk",
"latex-workshop.latex.autoBuild.run": "onSave",
"latex-workshop.view.pdf.viewer": "tab"
```

## Manual recipe (if you skipped `latexmk`)

```bash
cd AtomSim/dynamics/diff_drive/analysis
pdflatex -interaction=nonstopmode -halt-on-error diff_drive.tex
pdflatex -interaction=nonstopmode -halt-on-error diff_drive.tex   # second pass settles TOC + refs
```

Two passes is enough for this document. If `pdflatex` ends with `LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.`, run it once more.

`-interaction=nonstopmode` keeps `pdflatex` from dropping into its interactive prompt on a soft error; `-halt-on-error` makes it stop hard on a real error instead of skidding through. Use them together for sane output.

## Adding new section files

The main `diff_drive.tex` is a thin shell — preamble + a list of `\input{sections/NN_topic}` lines. To add a section:

1. Create `sections/NN_topic.tex` with a top-level `\section{…}\label{sec:topic}` and the body content. Don't include a preamble or `\begin{document}` — those live in the main file.
2. Add `\input{sections/NN_topic}` to the appropriate spot in `diff_drive.tex`.
3. Reuse the macros defined in `preamble.tex` (`\R`, `\xbody`, `\trackw`, `\taum`, `\half`) — extend that file when you need a new one rather than redefining locally.

Cross-references work across files automatically as long as labels are unique (`\label{eq:noslip}`, `\label{sec:constraints}`, etc.).

## Common errors

| Symptom | Cause |
|---|---|
| `! LaTeX Error: File 'sections/02_constraints.tex' not found` | You ran `pdflatex` from the wrong directory. `\input{}` paths are resolved relative to wherever you invoked the compiler. Always run from the dir containing the main `.tex`. |
| `! Undefined control sequence. \xbody` | You forgot `\input{preamble}` in the main file, or you're trying to compile a section file directly. Section files are not standalone — only the main file has `\documentclass`. |
| TOC is empty / shows old section names | First-pass artefact. Re-run the compile (`latexmk` does this automatically; manual `pdflatex` needs a second pass). |
| `! Package hyperref Warning: Token not allowed in a PDF string` | A LaTeX command appeared inside a section title or caption that hyperref can't render in the PDF outline. Wrap it in `\texorpdfstring{…}{…}` with the rendered form on the right. |
