MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86.sh
INSTALLATION_DIR=~/Miniconda3-4.3.14-Linux-x86
ENV_NAME=openml_study14

if [ ! -d "$INSTALLATION_DIR" ]; then
  wget "$MINICONDA_URL" -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$INSTALLATION_DIR"
fi

export PATH="$INSTALLATION_DIR/bin:$PATH"

conda env create -f environment.yml
source activate openml_study14

pip install xmltodict==0.11.0
pip install git+https://github.com/renatopp/liac-arff@6090cbadb489250db42a499d0094828c1e664ef2

# TODO install latest OpenML version

echo "###"
echo "conda list"
conda list