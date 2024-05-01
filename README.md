# Graph Neural Networks for Graph Classification Benchmark

This project aims to compare the interactions between convolutional, pooling and readout layers on a graph classification task.

![plot](./images/standard_archi.png)

This is a simplified version of an original research project done for the M2DS course : ML research seminar (https://github.com/AntoineTSP/ML_research_seminar/tree/main).

# Credentials

If you are not using an ssp cloud service, you might need to add your credentials in a .env file

# Download data

```
python data/download_data.py 
```

# To use training_script

```
python -m model.training_script -c configs/templates/MUTAG_test_template.yml
```

One has to properly fill the yml file. Don't forget to install the yaml module.

New layers of convolution or pooling have to be added to model\layer_selector.py.

New layers of local pooling should also contained a dictionary indicating the order of the variables returned by the forward of the pooling layer. For instance the MEWISPooling has the dictionary : 

```
{'node_features':0,'edge_index':1,'batch':2, 'loss':3}
```

While the SAGPooling has the dictionary :

```
{'node_features':0,'edge_index':1,'batch':3}
```

Since the forward method of SAGPooling returns an edge_attributes values at the second place (which is useless to us), and similarly the MEWISPooling returns a 
loss at the third place that has to be taken into account.

# Multiple models training

```
python main.py
```

One need to place the basic config templates inside the folder for each different dataset

# API

# Docker

```
docker build . -t gerardtho/gnn  
```

```
docker run -p 5000:5000 -it gerardtho/gnn 
```

# Project organization