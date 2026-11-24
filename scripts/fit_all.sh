#!/bin/bash
# All pooled fits behind the paper, one league configuration at a time.
#
# Outputs, per configuration NAME, in results/<sport>/:
#   AC_<NAME>_pooled_sym_em.npz        AC, constant-sum slopes, home boost      (Tables 1-3)
#   AC_<NAME>_pooled_sym_em_eta0.npz   same with eta=0                          (Table 2, Sec. 4.1)
#   CL_<NAME>_pooled_sym_em.npz        CL, same home boost and prior            (Table 2, Sec. 4.2)
#   AC_<NAME>_pooled_free_em.npz       free slopes (venue-dependent rule)       (Conclusions)
# Season-by-season fits (Sec. 4.1): python scripts/run_per_season.py
#
# Usage (from the repository root):
#   bash scripts/fit_all.sh            all configurations, sequentially; NHL_4 and SHL_4 take hours
#   bash scripts/fit_all.sh EPL NHL_4  selected configurations
# The data must be in data/<league>/ (see data/README.md).
set -e
cd "$(dirname "$0")/../src"
PY=${PY:-python3}
NAMES=( "$@" )
[ ${#NAMES[@]} -eq 0 ] && NAMES=( Division1_2pt Division1_3pt EPL Championship Bundesliga_2pt Bundesliga_3pt NHL_4 SHL_4 SuperLega )

for NAME in "${NAMES[@]}"; do
  echo "=== $NAME ==="; date
  $PY optimize_models.py --mode joint --name "$NAME" --model AC --method em --symmetric        --n-boot 0 --overwrite
  $PY optimize_models.py --mode joint --name "$NAME" --model AC --method em --symmetric --eta0 --n-boot 0 --overwrite
  $PY optimize_models.py --mode joint --name "$NAME" --model CL --method em --symmetric        --n-boot 0 --overwrite
  $PY optimize_models.py --mode joint --name "$NAME" --model AC --method em                    --n-boot 0 --overwrite
done
echo "=== Done ==="; date
