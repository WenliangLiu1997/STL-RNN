# STL-RNN
The implementation code for the two case studies in the paper "Recurrent neural network controllers for signal temporal logic specifications subject to safety constraints". Details can be found in the paper.
## Installation
You need Python3, Numpy, Scipy and Pytorch installed
## Usage
\# = 1 and 2 for case study I and case study II, respectively.

- Run *generate_dataset_case#.py* to generate the dataset. 
- Run *rnn_train_case#.py* to train the RNN.
- Run *rnn_test_case#.py* to test the RNN.

*Q_case#.npy* and *U_case#.npy* are the generated dataset. rnn_case#.pkl is the trained model.

The dataset and the trained model are both contained. So the easiest way to use the controller is directly run the rnn testing code. You can also use generate_dataset_casex.py to generate your own dataset or use rnn_train_casex.py to train your RNN model.

## Citation
When citing our work, please use the following BibTex:
'''
@article{liu2021recurrent,
  title={Recurrent neural network controllers for signal temporal logic specifications subject to safety constraints},
  author={Liu, Wenliang and Mehdipour, Noushin and Belta, Calin},
  journal={IEEE Control Systems Letters},
  year={2021},
  publisher={IEEE}
}
'''
