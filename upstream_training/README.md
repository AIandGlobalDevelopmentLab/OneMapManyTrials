# Upstream Training

This directory contains the scripts and data used to train the upstream model on a SLURM system. To train all the models used in the downstream analysis, you will need to run the script:

```
sbatch train_models_array.sh
```

changing the `PROJECT_CREDENTIALS` to your project credentials in the script. This will run the training jobs in parallel, each with a different set of hyperparameters as specified in the `train_models_array_args.txt` file. To train a single model, you can run the script:

```
sbatch train_model.sh
```