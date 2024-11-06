# PS 2, ex 2 Smoking is bad

## Title: Smoking is bad
## Credits: Leo Higgins and Charlie Warbuton
## Description: Project aims to disprove claim that smoking leads to higher chance of survival. May seem this way because of the dataset given, where there were far more observations of young people rather than old people who were all dead. Logistic regression suggests age contributes to survival more than smoking does.
## Dataset: 1300 observations of indiviudal people, with variables of their age, salary and whether they smoke and/or are alive

# Setup

To install the required packages, run the following command in the terminal (!You have to first
create the environment.yml!):

```bash
conda env create -f environment.yml
conda activate smoke
```

## Pre-commit

This repository uses pre-commit to run some checks before each commit. !You have to first create
the .pre-commit-config.yaml!

To install pre-commit, run:

```bash
pre-commit install
```

To run the checks manually, run:

```bash
pre-commit run --all-files
```
