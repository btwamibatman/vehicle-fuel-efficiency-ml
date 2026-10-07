# Contributing

## Ground rules

- Everyone commits **from their own GitHub account** (`git config user.name` / `user.email` set to yourself). Never commit on a teammate's behalf or reuse their identity.
- Never commit directly to `main`. One feature, one branch, one pull request.
- Edit only files you own (see [docs/team_tasks.md](docs/team_tasks.md)). If you need a change in someone else's file, ask them or open a small PR they review.

## Workflow

```bash
git checkout main && git pull
git checkout -b khamza/linear-regression      # <name>/<topic>
# ... work ...
pytest                                        # must pass
git add <specific files>
git commit -m "Add Linear Regression pipeline builder"
git push -u origin khamza/linear-regression
```
Then open a pull request on GitHub.

## Small commits

Commit when one logical thing works (one model, one plot, one doc section). Messages in imperative mood: "Add KNN builder with scaling". Don't commit `.venv`, raw data, or notebook outputs with huge cells.

## Pull requests

A PR description states: what changed, how you checked it (command + result), and anything the reviewer should look at. Keep PRs small enough to review in 10-15 minutes.

## Reviewing a teammate's work

1. Pull the branch, run `pip install -e ".[dev]"` and `pytest`.
2. Run what the PR claims (e.g. `python -m vehicle_efficiency.train --model knn`).
3. Check for **data leakage**: is anything fitted on validation/test data? Is the test set touched before the final model is chosen? Is scaling inside the Pipeline?
4. Check that interpretations match the plots/numbers and no results are invented.
5. Leave specific comments; approve only when you understand the change - you will have to explain it in the defense.

## Notebooks

Notebooks merge badly. Only one person edits a given notebook at a time; say so in chat. Put reusable code in `src/`, not in notebook cells. Use *Restart & Run All* before committing.

## Contribution record

After merging, add your real PR/commit link to the table in the README. Do not add entries for work you did not do.
