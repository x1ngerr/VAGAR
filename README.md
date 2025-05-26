# VAGAR: Variational Hypergraph Embedding with Adversarial Optimization for Predicting circRNA-miRNA Associations

## Project Structure



    VAGAR/
    ├── code/
    │   ├── main.py              # Main program entry
    │   ├── config.py            # Configuration parameters
    │   ├── data_utils.py        # Data loading and processing functions
    │   ├── hypergraph_utils.py  # Hypergraph construction tools
    │   ├── models.py            # Neural network models
    │   ├── metrics.py           # Evaluation metrics
    │   ├── train_utils.py       # Training and testing functions
    ├── data/                    # Data directory
    │   ├── CMI-9905/            # Dataset CMI-9905
    │   ├── CMI-9589/            # Dataset CMI-9589
    │   └── CMI-20208/           # Dataset CMI-20208
    └── README.md                # Project documentation


## Module Descriptions
### 1.config.py:
Defines command-line argument parser, including data paths, cross-validation folds, training epochs, and other hyperparameters.

### 2.data_utils.py: 
Provides functions for data loading, processing, and preprocessing, including reading CSV/TXT/MAT files, preparing training/testing datasets, and constructing input features.

### 3.hypergraph_utils.py: 
Contains tools for hypergraph construction and processing, implementing functionality to build high-order bipartite hypergraphs based on miRNA-disease association matrices.

### 4.models.py: 
Defines multiple neural network model classes, including:
HGNN_conv: Hypergraph convolutional layer
HGCN: Hypergraph convolutional network
VGAE: Variational graph autoencoder
GraphGAN: Graph generative adversarial network
VAGAR: Main model integrating hypergraph convolution and VGAE

### 5.metrics.py: 
Implements calculation of various evaluation metrics such as AUC, AUPR, F1 score, accuracy, sensitivity, specificity, etc.

### 6.train_utils.py: 
Contains training and testing functions.

### 7.main.py: 
Main program entry, coordinating the entire training and evaluation process, implementing cross-validation and result summarization.



## Running the Main Program

To run the program, simply execute the main.py file. This will initiate the full process of data loading, feature extraction, model training, and evaluation.

    python main.py

## Requirements

Ensure that the following Python packages are installed:

    pip install numpy keras scikit-learn joblib tqdm matplotlib

## Contributing

Contributions are welcome! If you find a bug or have a suggestion, please open an issue or submit a pull request.
