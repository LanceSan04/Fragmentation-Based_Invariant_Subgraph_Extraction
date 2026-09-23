> September 23 update: historical/generated folders were archived. Current locations and recovery instructions are in [CLEANUP.md](CLEANUP.md); older location statements below are historical.

# REU file locations

Organized September 19, 2026. All paths below are on the local Windows computer. Server files and commands were not moved.

## Where to look

- Downloaded result snapshots: `outputs/imported/downloads/` inside the REU repository.
- Older unreferenced local runs: `outputs/archive/local/`.
- Downloaded reports, poster, presentations and tutorials: `docs/reference/downloads/`.
- Server upload ZIP: `outputs/transfers/REU_Consolidation_Update.zip`.
- Newly launched experiments still use the consolidated main.py output routing.

Folders retain their original names. Snapshots were not merged, deleted, or deduplicated. An incomplete snapshot can still contain useful evidence.

## Kept in place

- `outputs_overnight_losses_20260913_102501`: preserve the processed-cache location.
- `outputs_diversity_schedule` and `outputs_policy_comparison`: legacy configs still name these output roots.
- `final_testing_outputs`, `logs`, `temp`, root report/audit/quickstart/license files: tracked historical or source material; relocation needs a separate reference review.
- `.venv`, `data`, source/config/tests, and unrelated Downloads were preserved.

## Verification and recovery

Move manifest: `C:/Users/Lance/Documents/GitHub/REU_Summer_2026/outputs/organization/20260919_020416/moves.json`.

Every moved item passed a before/after inventory comparison of relative file names, file sizes and modification times. This is an inventory check, not a cryptographic content hash. Moves stayed on the same local volume. Original Obsidian notes changed by link repair are backed up alongside the manifest. Saved experiment configs and metrics were not rewritten: paths inside those historical records may still describe their original server or local location. Use an explicit processed-dir override when evaluating a relocated cache.

## Original and current locations

| Original | Current |
|---|---|
| `C:\Users\Lance\Downloads\drugood_lbap_core_ec50_assay__mixed_group_graph1_brics1_recap1__full__20260716_185729` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\drugood_lbap_core_ec50_assay__mixed_group_graph1_brics1_recap1__full__20260716_185729` |
| `C:\Users\Lance\Downloads\drugood_lbap_core_ec50_assay_group_graph` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\drugood_lbap_core_ec50_assay_group_graph` |
| `C:\Users\Lance\Downloads\outputs` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs` |
| `C:\Users\Lance\Downloads\outputs_assay_erm_matched_5001_5003` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_assay_erm_matched_5001_5003` |
| `C:\Users\Lance\Downloads\outputs_dual_policy_assay` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_dual_policy_assay` |
| `C:\Users\Lance\Downloads\outputs_dual_policy_assay_20260917_150926_532008` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_dual_policy_assay_20260917_150926_532008` |
| `C:\Users\Lance\Downloads\outputs_dual_policy_assay_test2` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_dual_policy_assay_test2` |
| `C:\Users\Lance\Downloads\outputs_ec50_scaffold_benchmark` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_ec50_scaffold_benchmark` |
| `C:\Users\Lance\Downloads\outputs_ec50_size_benchmark` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_ec50_size_benchmark` |
| `C:\Users\Lance\Downloads\outputs_inference_scaffold` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_inference_scaffold` |
| `C:\Users\Lance\Downloads\outputs_inference_seed3001_3002` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_inference_seed3001_3002` |
| `C:\Users\Lance\Downloads\outputs_inference_seed3003` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_inference_seed3003` |
| `C:\Users\Lance\Downloads\outputs_inference_seed3004_3005` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_inference_seed3004_3005` |
| `C:\Users\Lance\Downloads\outputs_inference_size` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_inference_size` |
| `C:\Users\Lance\Downloads\outputs_multiview_ablation` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_multiview_ablation` |
| `C:\Users\Lance\Downloads\outputs_overnight` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_overnight` |
| `C:\Users\Lance\Downloads\outputs_overnight_losses_20260913_102501` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_overnight_losses_20260913_102501` |
| `C:\Users\Lance\Downloads\outputs_parameter_search_20260915_043519` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_parameter_search_20260915_043519` |
| `C:\Users\Lance\Downloads\outputs_parameter_search_size_20260916_135421` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_parameter_search_size_20260916_135421` |
| `C:\Users\Lance\Downloads\outputs_parameter_smoke_size_20260917_145934` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_parameter_smoke_size_20260917_145934` |
| `C:\Users\Lance\Downloads\outputs_parameter_winner_extra` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_parameter_winner_extra` |
| `C:\Users\Lance\Downloads\outputs_parameter_winner_test` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_parameter_winner_test` |
| `C:\Users\Lance\Downloads\outputs_policy_comparison` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_policy_comparison` |
| `C:\Users\Lance\Downloads\outputs_preliminary_test` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\outputs_preliminary_test` |
| `C:\Users\Lance\Downloads\overnight_and_diagnostics` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\imported\downloads\overnight_and_diagnostics` |
| `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs_dual_policy_assay_20260917_110337_356193` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\archive\local\outputs_dual_policy_assay_20260917_110337_356193` |
| `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs_dual_policy_assay_20260918_191201_959284` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\archive\local\outputs_dual_policy_assay_20260918_191201_959284` |
| `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs_parameter_smoke_size_20260916_095128` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\archive\local\outputs_parameter_smoke_size_20260916_095128` |
| `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs_parameter_smoke_size_20260917_102042` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\archive\local\outputs_parameter_smoke_size_20260917_102042` |
| `C:\Users\Lance\Downloads\AUDIT_AND_CHANGES.md` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\AUDIT_AND_CHANGES.md` |
| `C:\Users\Lance\Downloads\REU Progress Report Sept 13.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU Progress Report Sept 13.pdf` |
| `C:\Users\Lance\Downloads\REU Site Mid-Progress Report.docx` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU Site Mid-Progress Report.docx` |
| `C:\Users\Lance\Downloads\REU_Intro_to_Networks (1).pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Intro_to_Networks (1).pdf` |
| `C:\Users\Lance\Downloads\REU_Report.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Report.pdf` |
| `C:\Users\Lance\Downloads\reu_site_mid-progress_report.docx` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\reu_site_mid-progress_report.docx` |
| `C:\Users\Lance\Downloads\REU_Summer_2026_Poster (1).pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Summer_2026_Poster (1).pdf` |
| `C:\Users\Lance\Downloads\REU_Summer_2026_Poster.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Summer_2026_Poster.pdf` |
| `C:\Users\Lance\Downloads\REU_Tutorial1_Part1_ML_Madeeha.ipynb` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Tutorial1_Part1_ML_Madeeha.ipynb` |
| `C:\Users\Lance\Downloads\REU_Tutorial1_Part2_NN_Lilia.ipynb` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Tutorial1_Part2_NN_Lilia.ipynb` |
| `C:\Users\Lance\Downloads\REU_Tutorial3_GNN.ipynb` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REU_Tutorial3_GNN.ipynb` |
| `C:\Users\Lance\Downloads\REUPresentation.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REUPresentation.pdf` |
| `C:\Users\Lance\Downloads\REUPresentation.pptx` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\REUPresentation.pptx` |
| `C:\Users\Lance\Downloads\Study Guide for REU Delab.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\Study Guide for REU Delab.pdf` |
| `C:\Users\Lance\Downloads\SuGAr.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\SuGAr.pdf` |
| `C:\Users\Lance\Downloads\Training_&_Inference_Pipeline.pdf` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\Training_&_Inference_Pipeline.pdf` |
| `C:\Users\Lance\Downloads\Training_&_Inference_Pipeline.png` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\docs\reference\downloads\Training_&_Inference_Pipeline.png` |
| `C:\Users\Lance\Downloads\REU_Consolidation_Update.zip` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\transfers\REU_Consolidation_Update.zip` |
| `C:\Users\Lance\Downloads\launch_overnight.py` | `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\transfers\legacy_downloads\launch_overnight.py` |
