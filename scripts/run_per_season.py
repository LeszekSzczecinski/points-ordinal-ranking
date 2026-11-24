# Sec. 4.1 of the paper: season-by-season AC fits (NHL, SHL, SuperLega), saved to results/.
# Run (from the repository root): python scripts/run_per_season.py
import os
import sys, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import optimize_models as om
import jax
cfg = om.SEASON_CONFIG_JOINT
plan = [('NHL', 4, cfg['NHL_4']), ('SHL', 4, cfg['SHL_4']), ('SuperLega', 6, cfg['SuperLega'])]
for lg, rc, years in plan:
    for y in years:
        t0 = time.time()
        try:
            om.fit_season(lg, y, result_cat=rc, symmetric=True, method='em', model='AC',
                          save=True, overwrite=False, verbose=True)
            print(f'DONE {lg} {y} {time.time()-t0:.1f}s', flush=True)
        except Exception:
            print(f'FAIL {lg} {y}', flush=True); traceback.print_exc()
        jax.clear_caches()
print('ALL DONE', flush=True)
