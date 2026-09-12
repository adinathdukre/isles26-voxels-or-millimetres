Place **one** skull-stripped T1w volume here before running `../../../../../do_test_run.sh`.

`.mha`, `.nii.gz` and `.nii` are all accepted (`find_input_image` in
`inference.py` globs for them in that order, and takes the first match). Grand
Challenge always supplies a single `.mha` at this path; the extra extensions are
so you can drop an ISLES'26 training volume straight in, e.g.

    sub-XXXX_ses-1_space-orig_desc-brain_T1w.nii.gz

No image is shipped here — this repository contains no imaging data.
