MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh
INSTALLATION_DIR=~/Miniconda3-4.3.14-Linux-x86_64
ENV_NAME=openml_study14

if [ ! -d "$INSTALLATION_DIR" ]; then
  wget "$MINICONDA_URL" -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$INSTALLATION_DIR"
fi

export PATH="$INSTALLATION_DIR/bin:$PATH"

conda env create -f environment.yml
source activate openml_study14

pip install xmltodict==0.11.0 liac-arff==2.1.1 lockfile

# TODO install latest OpenML version

echo "###"
echo "conda list"
conda list
