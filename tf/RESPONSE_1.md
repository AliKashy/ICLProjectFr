# Response 1: the review of package 009

21 October 2026. This answers the review of `releases/009_20261021_180056_UTC.zip` point by point. The review found no defect in the package. Its main advice is right: personalize and install on the server, not on Windows.

## Two corrections to the review's install values

- **The SHA-256 ends in `8d09`, not `8de9`.** The full value is `e8520703d21769a6e8b2ab6185ce13dc2c0eddd74b79b0d29f6e68ba8e768d09`.
- **The package name in the review's step 3 is misspelled by one letter.** Use the exact name in the private runbook. A wrong name personalizes every file wrongly.
- Upper or lower case for the two ids makes no difference. `personalize.py` upper-cases and lower-cases them itself.

## The six points

1. **Windows line endings.** True. On Windows, `personalize.py` writes `PACKAGE.json` and `PERSONALIZATION.json` with CRLF line endings. Every code file is still written byte for byte, so the install works. Personalizing on the server avoids the question entirely, so that is the route to use.
2. **Do not upload local code.** Agreed. Upload only the package ZIP. The install receipt compares the server's own V1 files.
3. **No runbook in the ZIP.** By design. Documents stay on GitHub and are never installed on the server. Every command comes from the runbook.
4. **No tests in the ZIP.** By design. The server package carries no tests and no synthetic data. The tests run before each release: 521 passed and 1 skipped under each of pandas 2.3.3 and 3.0.5.
5. **`???` in `c05.yaml` and `c20.yaml`.** By design. Step 1 measures those values, and the exact settings are given before those fits.
6. **xgboost, scikit-learn and joblib.** Step 0's environment check lists any missing library. They are needed only from the feature screen and step 4 onward.

## The install, on the server only

The exact lines, with the real values, are in the private runbook, section 0, "Only if the laptop has no Python". In outline:

1. Upload the original `009_20261021_180056_UTC.zip` to the project folder on the server.
2. Check it. The line must print the value above:

```bash
cd <project folder> && sha256sum 009_20261021_180056_UTC.zip
```

3. Only if it matches, extract, personalize and install:

```bash
cd <project folder>
python -m zipfile -e 009_20261021_180056_UTC.zip .tf_install &&
$PY .tf_install/009_20261021_180056_UTC/personalize.py --package <package name> --id <id> --reference-id <reference id> --data-bucket <data bucket> --artifact-bucket <artifact bucket> &&
$PY .tf_install/009_20261021_180056_UTC/install.py --target <project folder> &&
rm -r -- .tf_install/009_20261021_180056_UTC &&
rmdir -- .tf_install
```

It prints `Configured: 009_20261021_180056_UTC`, then `Installed and verified: 009_20261021_180056_UTC`. If a line stops, run `rm -r -- .tf_install` and paste the block again. `$PY` is the Python of the environment the version 1 runs used.

4. Then step 0, the install check, from the runbook. It is read-only and needs no GPU.

Do not edit, reformat or save any package file at any point. `personalize.py` and `install.py` refuse a changed file. That refusal is safe, but the steps must then start again.
