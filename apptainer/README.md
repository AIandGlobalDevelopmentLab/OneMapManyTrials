# Apptainer

For the sake of reproducibility, the whole project was conducted within the same Apptainer image. Apptainer is an open‑source container platform designed for high‑performance computing (HPC) and scientific workflows, allowing users to run applications in portable, reproducible environments.

The Apptainer image (`.sif`) used for this project is too large to be stored directly on GitHub. We are happy to provide it upon request.

Alternatively, you can build the image yourself using the included `recipe.def` to reproduce the same environment. To build the image, run:

```bash
apptainer build image.sif recipe.def
```

This will create a `image.sif` file in your current directory that you can use to run the project in a fully reproducible environment.
