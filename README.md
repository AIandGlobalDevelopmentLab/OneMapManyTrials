# One Map, Many Trials

To try out estimating an ATE using our pipeline on simulated data, run the script `simulations/run_simulation` with your selected parameters.

For running multiple simulation rounds, e.g. in order to validate the empirical coverage of the confidence intervals, you can use the SLURM job-array script `simulations/run_ate_array.sh`. The job array accepts script accepts the same arguments, e.g.

`sbatch simulations/run_ate_array.sh --sigma_X=3.0 --n_trial_samples=20000 --tau=0.5`

Remember to first set up your configuration file `config.ini`, following the example in `config-sample.ini`.
