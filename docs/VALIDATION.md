# Current validation after the reviewer consistency audit

The candidate ZIP was extracted to a new directory containing a space and run from outside that directory with the isolated locked environment. All 13 stages and 86 CSV comparisons passed in 30.708 seconds. Maximum absolute numeric difference was 3.46e-15; integer and Boolean columns are exact and frozen input hashes are unchanged. All 27 table snapshots are exported.

The binary partial-integral output now includes the nominal 0.50 endpoint, matching its declared interval. Only two summary columns differ from the prior version; the complete risk curves, AURCs and selections are unchanged. All nine main-figure plotted-data CSV files are byte-identical to the previous verified run. See CONSISTENCY_CORRECTIONS.md. The accompanying manuscript, supplementary guide and 23 reviewer responses were reconciled separately. All original table cells, native equations and embedded supplementary images are preserved. DOCX pagination remains unverified because the configured renderer lacks soffice.exe. No new model training, model inference, public release or resubmission occurred.

# Earlier validation after the fixed batch budget revision

Validated on Windows x64 on 28 September 2026 using the isolated locked environment. All 13 pipeline stages and 86 reference CSV comparisons passed in 30.812 seconds; maximum absolute numeric difference 3.46e-15. Integer and Boolean columns are exact; frozen input hashes are unchanged. The candidate ZIP was extracted to a new directory containing a space and invoked from outside that directory. Twenty-seven reviewed table snapshots are exported (6 main, 21 supplementary).

The new batch comparison includes all 27 runs for review volume and original tile-mean loss; 21 runs with frozen confusion counts additionally supply pooled risk and content. The fixed policy reviews floor(N/5) tiles with stable original-position tie handling. Independent reverse sorting verifies all 54 policy–run selections and all original threshold counts. The largest independent tile-loss mean difference is 5.56e-17. E2/E3 content is unavailable and is not imputed. See FIXED_BUDGET_PROTOCOL.md for data provenance and the audit-versus-original tile-loss sensitivity check.

The added Table S21 PDF page was visually inspected. All 27 prior table-PDF page content streams remain byte-identical. Existing manuscript tables, nine native equations and seven supplementary images are unchanged. DOCX rendering still fails because the configured environment has no soffice.exe; final Word pagination is not verified. No model inference, training, human-review evaluation or public release occurred in this revision.

# Earlier validation after the multiclass revision

Validated on Windows x64 on 28 September 2026 using the same isolated environment and dependency lock. The updated package was zipped, extracted to a new directory containing a space, and invoked from outside that directory. All 12 stages completed in 31.559 seconds. All 81 reference CSV comparisons passed; the maximum absolute numeric difference was 3.46e-15. Integer and Boolean columns remain exact, and the input hashes were unchanged. Twenty-six reviewed table snapshots are exported, including Table S20.

The new scope audit independently reconstructs both class conventions from frozen confusion counts. It matches the original full AURCs within 1.12e-16 and the original event curves within 1e-12. Six seed–ordering results and 60 event–seed–ordering results are retained. The nominal 0.50 endpoint is now included in the partial integral: 83.3% applies to 0.10–0.50, whereas 78.6% applied to 0.10–0.45. Full curves and selections are unchanged. Figure 5 derives its percentage from the curve, rather than a literal annotation.

The corrected Figure 5 (color and grayscale) and the added Table S20 PDF page were visually inspected. The previous 26 independent table-PDF page content streams remain byte-identical. This is separate from DOCX pagination, for which the configured renderer still lacks soffice.exe. Source-data checks, native-math/media preservation and numeric checks passed; final Word pagination is unverified.

Current machine-readable reports are `validation_reference.json` and `reproduction_reference.json`; previous reports are retained under filenames ending in `before_multiclass_correction.json`. The current package has 28 Python source files. Optional model-inference validation remains limited to the previously checked command-line imports; no new model inference or training was performed.

# Earlier validation before the multiclass interval correction

Validated on Windows x64, 28 September 2026. A newly created virtual environment installed the packages recorded in `requirements-lock.txt`. The package was zipped, extracted to a different directory containing a space, and invoked from a working directory outside the extracted package. No original study directory was required by the CPU pipeline.

| Check | Result |
|---|---|
| One-command pipeline | All 11 stages passed; 29.912 seconds on this machine |
| Frozen reference comparison | 78/78 CSVs passed; maximum absolute numeric difference 3.4555691641457997e-15 |
| Integer/Boolean output columns | Exact equality required and passed |
| Primary calibrated coefficient/score reproduction | Maximum difference 1.6028844918025698e-15; absolute fit bound 1e-12 |
| Primary calibrated ranking and retained sets | Exact equality across 18 runs |
| Calibrated primitive replay values and comparator AURC checks | Zero difference in the temperature analysis cross-checks |
| Input files | Hashes unchanged by the full pipeline |
| Parquet export | 45/45 score tables passed exact numeric and record-order checks at assembly |
| Figures | Five one-page PDFs with embedded fonts; five RGB TIFFs at 600 dpi; layout checks passed |
| Visual inspection | Color and grayscale PDF-derived renders inspected; no observed clipping or text collisions after adjusting Figure 1 count labels for DejaVu Sans |
| Tables | 25 reviewed table snapshots exported; coverage of numerical recomputation documented separately |
| Python sources | 27 source files compiled successfully |
| Optional model commands | Both replay `--help` commands passed in the observed model environment; a GDAL_DATA warning was emitted; no new inference performed |

Machine-readable evidence is supplied in `validation_reference.json` and `reproduction_reference.json`. These describe the verified analysis code and input files; final README/validation documentation was completed after the test. The final archive contains a SHA-256 manifest.

## Portability changes

Absolute author-machine paths were replaced with package-relative inputs and explicit output/data-root options. Frozen Parquet scores were exported to CSV without numeric changes. Internal editorial report generation was removed from two summary scripts while their numeric output stages were preserved. Matplotlib's bundled DejaVu Sans replaces the submission figure font; Figure 1 count labels use two lines to avoid collisions. Numerical figure data remain unchanged within the stated checks.

An initial fresh-environment run stopped at an original bitwise float-equality assertion in coefficient fitting. The discrepancy was measured across all 18 runs before changing the check. The portable script records the error, requires a 1e-12 absolute bound, and additionally enforces exact original rank order and calibrated retained sets. It does not round scores, replace fitted values with saved values, or relax integer counts.

This record establishes local calculation reproduction from bundled frozen inputs. It does not establish fresh model training, independent data validation, portable GPU inference, final manuscript pagination, or completion of public release.
