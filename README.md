there are four folders in this project
/data           （sava the data and some config files）        
/save           （save the model paramter and some template model files）
/work-metrla     （work with METR-LA dataset）
/work-pemsbay    （work with PEMS-BAY dataset）


run with different dataset by working in two folder
cd work-metrla 
or 
cd work-pemsbay

# run STGCN
python runstgcn.py

# run TGCN
python runtgcn.py

# run Graph WaveNet
python rungraphwavenet.py

# run DCRNN
python rundcrnn.py



# configuration instructions：

## 1、enviroment：
python=3.6.6
tensorflow=1.14
pytorch=1.3.1
scipy=1.5.2
numpy=1.19.1
pandas>=0.19.2
tables
statsmodels
future
pyyaml
matplotlib

## config files of DCRNN:

run with metr-la:
/home/cseadmin/yindu/github/data/dcrnn-data/model/dcrnn_la.yaml

run with pems-bay:
/home/cseadmin/yindu/github/data/dcrnn-data/model/dcrnn_bay.yaml

test with metr-la::
/home/cseadmin/yindu/github/data/dcrnn-data/model/dcrnn_metrla_test_config.yaml

test with pems-bay:
/home/cseadmin/yindu/github/data/dcrnn-data/model/dcrnn_pemrbay_test_config.yaml


