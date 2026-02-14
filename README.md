# FIRE 2025 Dravidian Language Detection

This repository contains the code for our FIRE 2025 submission.

## ⚠️ Important Note on Published Paper

Our published paper contains inaccuracies in the hyperparameter 
descriptions (Section 4.1). The **actual** implementation used:

- Training epochs: **5** (not 10 as stated)
- Batch size: **8** (default, not 32 as stated)  
- Number of labels: **7-8** (not 4-5 as stated)

The results in Table 2 are correct and were generated using the
code in this repository with the above hyperparameters.

## Results
- 🥇 1st place: Tamil, Malayalam, Telugu
- 🥈 2nd place: Tulu
