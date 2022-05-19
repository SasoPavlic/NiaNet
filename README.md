<p align="center"><img src=".github/NiaNetLogo.png" alt="NiaPy" title="NiaNet"/></p>

---

### Designing and constructing neural network topologies using nature-inspired algorithms

### Description 📝

The proposed method NiaNet attempts to pick hyperparameters and AE architecture that will result in a successful encoding and decoding (minimal difference between input and output). NiaNet uses the collection of algorithms available in the library [NiaPy](https://github.com/NiaOrg/NiaPy) to navigate efficiently in waste search-space.

### What it can do? 👀

* **Construct novel AE's architecture** using nature-inspired algorithms.
* It can be utilized for **any kind of dataset**, which has **numerical** values.

### Installation ✅

Installing NiaNet with pip3: 
```sh
pip3 install nianet
```

### Documentation 📘

The paper referring to this source code is currently being published. The link will be posted here once it is available.

### Examples

Usage examples can be found [here](examples).

### Getting started 🔨

##### Create your own example
In [examples](examples) folder create the Python file based on the existing [evolve_for_diabetes_dataset.py](examples/evolve_for_diabetes_dataset.py).

##### Change dataset
Change the dataset import function as follows:
```python
from sklearn.datasets import load_diabetes
data = load_diabetes()
```

##### Specifying the Search space

Set the boundaries of your search space with [autoencoder.py](nianet/autoencoder.py).

The following dimensions can be changed:
* Topology shape (symmetrical, asymmetrical)
* Size of input, hidden and output layers
* Number of hidden layers
* Number of neurons in hidden layers
* Activation functions
* Number of epochs
* Learning rate
* Optimizer

You can run the NiaNet script once your setup is complete.
##### Running NiaNet script

`python evolve_for_diabetes_dataset.py`

### HELP ⚠️

**saso.pavlic@student.um.si**

## Acknowledgments 🎓

* NiaNet was developed under the supervision
  of [doc. dr Iztok Fister ml.](http://www.iztok-jr-fister.eu/)
  at [University of Maribor](https://www.um.si/en/home-page/).

* This code is a fork of [NiaPy](https://github.com/NiaOrg/NiaPy). I am grateful that the authors chose to
  open-source their work for future use.

## License

This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
