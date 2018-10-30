# joint-embedding-nmt

This repository contains a Pytorch implementation of the structure-aware output layer for
neural machine translation which was presented at WMT 2018
(<a href="http://publications.idiap.ch/downloads/papers/2018/Pappas_WMT_2018.pdf">PDF</a>).
The model is a  generalized form of weight tying which shares parameters between input and
output embeddings but allows learning a more flexible relationship with input word
embeddings and enables the effective capacity of the output layer to be controlled. In
addition, the model shares weights across output classifiers and translation contexts
which allows it to better leverage prior knowledge about them. 


```
@inproceedings{Pappas_WMT_2018,
  author    = {Pappas, Nikolaos and Miculicich, Lesly and Henderson, James},
  title     = {Beyond Weight Tying: Learning Joint Input-Output Embeddings for Neural Machine Translation},
  booktitle = {Proceedings of the Third Conference on Machine Translation (WMT)}, 
  address   = {Brussels, Belgium}, 
  year      = {2018}
}
```


## Source files

The files that are specific to our paper are the following ones: (i) GlobalAttention.py
modifies the original attention to support different input and output dimensions for the
attention mechanism, (ii) Generator.py contains the implementation of the proposed output
layer, and (iii) train.py the sampling-based training approach.

> - onmt/modules/GlobalAttention.py	
> - onmt/Generator.py
> - train.py            


## Dependencies

The available code is largely based on an earlier version (v0.2.1) of OpenNMT
(<a href="https://github.com/OpenNMT/OpenNMT-py" target="_blank">https://github.com/OpenNMT/OpenNMT-py</a>)
which requires Python (<a href=" http://www.python.org/getit/" target="_blank"> http://www.python.org/getit/</a>)
and Pytorch library (<a href=" https://pytorch.org/" target="_blank"> https://pytorch.org/</a>)
in order to run. For detailed instructions on how to install and use them please refer to
the corresponding links above.
 
 
## Contact

npappas@idiap.ch

