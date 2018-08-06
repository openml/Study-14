MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
WORKSPACE=/work/ws/nemo/fr_mf1066-openml-study99-0
if [ -d "$WORKSPACE" ]; then
    INSTALLATION_DIR="$WORKSPACE"/Miniconda3-4.5.4-Linux-x86_64
else
    INSTALLATION_DIR=~/Miniconda3-4.5.4-Linux-x86_64
fi
ENV_NAME=openml_study99

if [ ! -d "$INSTALLATION_DIR" ]; then
  wget "$MINICONDA_URL" -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$INSTALLATION_DIR"
fi

source "$INSTALLATION_DIR""/etc/profile.d/conda.sh"

conda activate
# conda create -n openml_study99 -y
# conda install numpy scipy nose requests scikit-learn==0.20.0 nbformat python-dateutil python cython -y
conda env create -f environment.yml
# TODO remove this installation command once the latest scikit-learn version is released!
pip install git+https://github.com/scikit-learn/scikit-learn
conda activate openml_study99

pip install babel==2.6.0 debtcollector==1.19.0 fasteners==0.14.1 \
    iso8601==0.1.12 liac-arff==2.2.2 monotonic==1.5 netaddr==0.7.19 \
    netifaces==0.10.7 oslo.concurrency==3.27.0 oslo.config==6.2.1 \
    oslo.i18n==3.20.0 oslo.utils==3.36.2 pbr==4.0.4 pyparsing==2.2.0 \
    pytz==2018.4 pyyaml==3.12 rfc3986==1.1.0 stevedore==1.28.0 \
    wrapt==1.10.11 xmltodict==0.11.0 xgboost==0.72.1

pip install git+https://github.com/openml/openml-python@666d4c78e96cb85a4c1e1074817ef835d6ca4e37
# conda env export

echo "###"
echo "conda list"
conda list
