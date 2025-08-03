# Preprocessing

This directory includes all the scripts required for downloading and preprocessing the data required for the analysis.

## DHS data

This project relies on the [DHSHarmonisation package](https://bitbucket.org/hansekbrand/dhsharmonisation/src/master/) for retrieving survey data. In order to get access to the surveys, you will need an authorised account registered with [the DHS program](https://dhsprogram.com/data/dataset_admin/login_main.cfm). These credentials must be added to the `config.ini` file, as described in the main README.

> ⚠️ **Important Notice**
> 
> Due to the current government freeze of the DHS Program, the organization has **paused the creation of new user accounts**. This means that if you do not already have an authorized DHS account, you will not be able to register for one and therefore cannot download the survey data through this workflow.
>
> Additionally, due to licensing restrictions, we are unable to share the exported DHS datasets directly. Redistribution of these datasets is prohibited by the DHS Program.

You then run:

```
Rscript 0_download_dhs_data.R
```

This will store the "raw" DHS dataset with household-level information in the data directory you specified in `config.ini`. The analysis in this paper is ran at the "cluster-level," roughly corresponding to a village in rural areas or a neighborhood in urban areas. In order to clean and group the raw dataset at the cluster-level, you need to run:

```
python 1_process_dhs_data.py
```

This results in the file `dhs_data.csv` in your data directory.

## Landsat data

To download the satellite data, run the notebook `2_landsat_exporter.ipynb`. The resulting dataset (about 32 GB in size) will be stored in the data directory as a `.np` file for each DHS cluster. Some of the cluster locations will not have any good satellite images (mostly due to clouding and poor coverage in the 90s), so we drop these in the `3_find_bad_images.ipynb` notebook before proceeding with the analysis. The resulting CSV file, `dhs_with_imgs.csv` will contain the clusters used in the experiments.

## HDF5 files

In order to speed up the training pipeline, we convert the dataset into the HDF5 format. This will require an additional 33 GB of storage, but greatly speeds up processing times. This is done byt running the script `4_write_hdf5.sh`.