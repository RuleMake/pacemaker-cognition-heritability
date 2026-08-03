#!/usr/bin/env bash
# Print the argument signatures of the individual gsMap steps.
#
# quick_mode has no switch to skip the report stage, and that stage dominates the
# runtime: it renders a few hundred PNGs per trait. Section 1 reached its Cauchy
# results in ~7 minutes and then spent 45 more inside report. Running the five
# analysis steps directly would cut an 8-section batch from ~7 hours to ~1.
source "$HOME/cardio/venv/bin/activate"
for c in run_latent_to_gene run_generate_ldscore run_spatial_ldsc run_cauchy_combination; do
  echo "===== $c"
  gsmap "$c" --help 2>&1 | grep -E '^\s+--' | sed 's/^/  /' | head -24
  echo
done
