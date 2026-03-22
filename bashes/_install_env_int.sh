export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh \
  /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml

source /scratch-ssd/oatml/miniconda3/bin/activate llm