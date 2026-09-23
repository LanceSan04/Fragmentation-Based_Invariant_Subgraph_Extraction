# Size contrastive sweep

Four new coefficients: 0, 0.01, 0.03, 0.3. Existing coefficient 0.1 is the reference; do not rerun it. Seeds7001–7010, 100 epochs, frozen Size settings, diversity/IRM zero, both checkpoint selection and inference policies. 40 new training runs. Test reporting is opt-in and exploratory; rank coefficients using validation, not test results.

Upload the partial REU_Size_Contrastive_Sweep_Update.zip into the existing server repository preserving directories. No source training implementation changes.

CUDA smoke:

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python main.py experiment contrastive-sweep --smoke --device cuda --evaluate-test
```

Full batch inside tmux:

```bash
tmux new -s size_contrastive
```

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python -u main.py experiment contrastive-sweep --dataset size --device cuda --evaluate-test --output-dir outputs/size/contrastive_sweep/seeds7001_7010
```

Detach Ctrl+B then D. Reconnect tmux attach -t size_contrastive. Run once. Fixed full output refuses overwrite. No automatic resume; preserve partial results and obtain recovery instructions on failure. May take multiple nights. Logs stream to terminal and per-condition .log files. Each condition has summary.json and per-seed configs/history/results/checkpoints. complete.json appears only after all conditions finish. Uses the existing data/processed/ec50_followup_v1/size cache.

No test reporting unless --evaluate-test explicitly supplied. The supplied full command opts in for the same four-way descriptive test comparison as prior runs. This is not an untouched-test confirmation of a newly selected coefficient.
