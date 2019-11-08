# Running the python experiments for study 99

## Installation

    git clone https://mfeurer@bitbucket.org/bernd_bischl/2016_openml_online_benchmark_paper.git study99
    cd study99/Study-14/PythonV2
    bash setup_conda.sh
    
## Generate Commands

    python /misc/generate_cluster_task_files.py --output-file /work/ws/nemo/fr_mf1066-openml-study99-0/commands.txt --run-tmp-dir /work/ws/nemo/fr_mf1066-openml-study99-0/run_cache/

And now run them all!

    while read LINE
    do
        sem -j 1 --id openml $LINE
    done < "/tmp/commands.txt"

## Dask command?

```bash
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
dask-scheduler --idle-timeout 600 --scheduler-file ~/.dask/scheduler_file
dask-worker --nthreads 1 --nprocs 1 --memory-limit 4GB --death-timeout 600 \
    --lifetime 600 --scheduler-file ~/.dask/scheduler_file --local-directory ~/.dask

``` 
