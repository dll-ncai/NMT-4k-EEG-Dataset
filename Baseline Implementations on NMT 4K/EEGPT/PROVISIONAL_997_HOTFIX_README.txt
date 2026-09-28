EEGPT-NMT v2: provisional 997-record evaluation hotfix
======================================================

Why the earlier command failed
------------------------------
The original eegpt_nmt/evaluate.py supported only --confirm-final-test.
The provisional flags did not yet exist, so argparse correctly reported them
as unrecognized. This hotfix implements those flags while retaining the strict
1,000-record official evaluation guard.

Installation on Windows
-----------------------
1. Extract this ZIP directly into:

       E:\EEGPT-NMT-v2

2. Allow Windows to merge folders and overwrite eegpt_nmt\evaluate.py.
   The new scripts\05a_provisional_evaluate_997.bat file will be added.

3. Open Anaconda Prompt or Command Prompt, then run:

       cd /d E:\EEGPT-NMT-v2
       conda activate eegpt
       python -m py_compile eegpt_nmt\evaluate.py
       python -m eegpt_nmt.evaluate --help

   The help output must show both:

       --confirm-provisional-incomplete-test
       --expected-available-records

Running the provisional evaluation
----------------------------------
From the project root, run:

       scripts\05a_provisional_evaluate_997.bat

Equivalent direct command:

       python -m eegpt_nmt.evaluate --config configs/default.yaml --confirm-provisional-incomplete-test --expected-available-records 997

Do not append .py after eegpt_nmt.evaluate when using python -m.

Output location
---------------
The provisional results are isolated under:

       outputs\final_evaluation\provisional_incomplete_cohort\
       eegpt_nmt_v2_attention_seed2026_997of1000\

Important safeguards
--------------------
- The checkpoint and threshold remain the validation-selected values.
- The evaluator does not tune anything on the 997 evaluation labels.
- The JSON records the available class counts and missing recording IDs.
- The output is explicitly marked provisional and is not the official
  1,000-record NMT-4K benchmark.
- scripts\05_final_evaluate.bat remains unchanged and still requires exactly
  1,000 evaluation recordings.
