# Mixture Models Library

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the Mixture Models Library repository! This repository contains a Python library that provides implementations and utilities for working with mixture models.

## Features

- **Gaussian Mixture Models (GMMs):** Includes methods for fitting Gaussian mixture models to data, estimating parameters, and performing probabilistic inference.

## Installation

You can install the library using pip:

```shell
pip install MixtureModels
```

## Getting Started

To get started with the library, refer to the [documentation](https://github.com/your-username/mixture-models-library/docs) for detailed usage examples, API reference, and tutorials.

```python
from MixtureModels import GaussianMixtureModel

# Fit a Gaussian Mixture Model to data
gmm = GaussianMixtureModel(n_components=3)
gmm.fit(data)

# Perform inference on new data points
probabilities = gmm.predict_proba(new_data)
```

## Contributing

Contributions are welcome! If you have any ideas, bug reports, or feature requests, please submit an issue or a pull request following our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this library according to the terms of the license.

## Acknowledgments

We would like to thank the contributors for their valuable contributions to this project.

If you find this library helpful in your research or work, please consider citing it:

```
@misc{mixturemodelslibrary,
  author = {Felipe Tufaile},
  title = {Mixture Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/MixtureModels}},
}
```

Thank you for using the Mixture Models Library! We hope it helps you in your data analysis and modeling tasks.
