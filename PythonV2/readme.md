# Running the python experiments for study 99

## Installation

    git clone https://mfeurer@bitbucket.org/bernd_bischl/2016_openml_online_benchmark_paper.git study99
    cd study99/Study-14/PythonV2
    bash setup_conda.sh
    
## Generate Commands

    python /misc/generate_cluster_task_files.py --output-file /work/ws/nemo/fr_mf1066-openml-study99-0/commands.txt --run-tmp-dir /work/ws/nemo/fr_mf1066-openml-study99-0/run_cache/

And now run them all!

    n_lines=`wc -l commands.txt | cut -f 1 -d ' '`
    for i in `seq 1 1 $n_lines`
    do 
        cmd=`head commands.txt -n $i | tail -1`
        echo $cmd
        exec $cmd
    done