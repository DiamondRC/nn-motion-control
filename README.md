[![CI](https://github.com/DiamondRC/deltabot-nn-controller/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamondRC/deltabot-nn-controller/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiamondRC/deltabot-nn-controller/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamondRC/deltabot-nn-controller)
[![PyPI](https://img.shields.io/pypi/v/deltabot-nn-controller.svg)](https://pypi.org/project/deltabot-nn-controller)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# deltabot_nn_controller

A Neural Network to control the motion of the Deltabot stage.

This is where you should write a short paragraph that describes what your module does,
how it does it, and why people should use it.

Source          | <https://github.com/DiamondRC/deltabot-nn-controller>
:---:           | :---:
PyPI            | `pip install deltabot-nn-controller`
Documentation   | <https://diamondrc.github.io/deltabot-nn-controller>
Releases        | <https://github.com/DiamondRC/deltabot-nn-controller/releases>

This is where you should put some images or code snippets that illustrate
some relevant examples. If it is a library then you might put some
introductory code here:

```python
from deltabot_nn_controller import __version__

print(f"Hello deltabot_nn_controller {__version__}")
```

Or if it is a commandline tool then you might put some example commands here:

```
python -m deltabot_nn_controller --version
```

<!-- README only content. Anything below this line won't be included in index.md -->

See https://diamondrc.github.io/deltabot-nn-controller for more detailed documentation.


# Required Setup

`uv venv --seed --system-site-packages`
`uv run python /workspaces/deltabot_nn_controller/src/deltabot_nn_controller/model.py`
