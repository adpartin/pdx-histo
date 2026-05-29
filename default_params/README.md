# default_params/

Hyperparameter templates, one per modality combination (`tile`, `ge`, `tile_ge`, `tile_dd1_dd2`, `ge_dd1_dd2`, `tile_ge_dd1_dd2`).

On first run for a given project, `src/trn_multimodal.py:158-161` selects the template matching the active `--use_*` flags, copies it to `projects/<prjname>/params_<feacombo>.json`, and uses the per-project copy from then on. Edit the per-project copy to change HPs without affecting templates here. See `PAPER.md` for the HPs that produced the published paper.
