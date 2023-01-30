<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/pierrehoudayer/RUBIS">
    <img src="Logo/RUBIS_logo_filtre.png" alt="Logo" width="400">
  </a>

<h3 align="center">Rotation code Using Barotropy conservation over Isopotential Surfaces (RUBIS)</h3>

  <p align="center">
    Write project description.
    <br />
    <a href="https://github.com/pierrehoudayer/RUBIS"><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="https://github.com/pierrehoudayer/RUBIS">View Demo</a>
    ·
    <a href="https://github.com/pierrehoudayer/RUBIS/issues">Report Bug</a>
    ·
    <a href="https://github.com/pierrehoudayer/RUBIS/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#deformation-method">Deformation Method</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
    <li><a href="#citing-rubis">Citing RUBIS</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

`RUBIS` (standing for *Rotation code Using Barotropy conservation over Isopotential Surfaces*) is a centrifugal deformation program that takes as input a 1D model (with spherical symmetry) and returns its deformed version by applying a conservative rotation profile specified by the user. More specifically, the code only needs the density as a function of radial distance, $\rho(r)$, from the reference model in addition to the surface pressure to be imposed, $P_0$, in order to perform the deformation. This lightness is made possible by the central procedure assumption which consists in preserving the relation between density and pressure when going from the 1D to the 2D structure. The latter makes it possible, in particular, to avoid the standard complications arising from the conservation of energy in the resulting model ([Jackson (1970)](https://ui.adsabs.harvard.edu/abs/1970ApJ...161..579J), [Roxburgh (2004)](https://doi.org/10.1051/0004-6361:20041202), [Jackson et al. (2005)](https://iopscience.iop.org/article/10.1086/426587), [MacDregor et al. (2007)](https://iopscience.iop.org/article/10.1086/518303)). In this sense, the method is analogous to the one presented by [Roxburgh et al. (2006)](https://doi.org/10.1051/0004-6361:20065109), but simpler since it does not require the calculation of the first adiabatic exponent, $\Gamma_1$, during the deformation and thus the explicit specification of an equation of state. 

As a result, the only equation effectively solved by the program is Poisson's equation, $\Delta \Phi = 4\pi G \rho$, leading to a very fast deformation of the model, even when high angular accuracies are required. Another feature of the method is its excellent stability, which allows for the deformation of models at speeds very close to the critical rotation rate (cf. figures [1][plot-example-1] & [2][plot-example-2] below). Finally, the code has been designed to allow both stellar and planetary models to be deformed, thereby dealing with potential discontinuities in the density profile. This is made possible by solving Poisson's equation in spheroidal rather than spherical coordinates whenever a discontinuity is present. More details regarding the <a href="#deformation-method">Deformation Method</a> can be found below.

| ![Example 1][plot-example-1] | 
|:--:| 
| Deformation of a polytropic structure with index $N=3$ at 99.99% of the critical rotation rate. Isopotentials are shown on the left and the density distribution on the right |
  
| ![Example 2][plot-example-2] | 
|:--:| 
| Idem for a $N=1$ polytrope.  |

Regarding the program itself, `RUBIS` is fully witten in `Python` since `v0.1.0`.


<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- GETTING STARTED -->
## Getting Started

Get a local copy of `RUBIS` by following the following steps.

### Prerequisites

`RUBIS` has been written in such a way that it only depends on the standard Python libraries: [NumPy][numpy-url], [SciPy][scipy-url] and [Matplotlib][matplotlib-url]. The `setup.py` is only used to ensure that these libraries are up to date.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/pierrehoudayer/RUBIS.git
   ```
2. Make sure that the standard libraries are up to date by running the `setup.py`
   ```sh
   python setup.py install
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- DEFORMATION METHOD -->
## Deformation Method

| ![Method][flowchart] | 
|:--:| 
| Flowchart illustrating how the model deformation method works. Each step shows the quantity that is obtained at the end of the procedure and in terms of which variable it is obtained. |


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

Give an example

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

List the forthcomming features.

See the [open issues](https://github.com/pierrehoudayer/RUBIS/issues) for a list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

`RUBIS` is in stage of early development. If you have a suggestion to improve this repository, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Any contribution is welcome and **greatly appreciated**!

1. Fork the Project
2. Create a Feature Branch (`git checkout -b feature-feature_description`)
3. Commit your Changes (`git commit -m 'Add some feature description'`)
4. Push to the Branch (`git push origin feature-feature_description`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- 
LICENSE
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->



<!-- CONTACT -->
## Contact

Pierre Houdayer -  pierre.houdayer@obspm.fr

[![ORCID][ORCID-shield]][ORCID-url]

Project Link: [https://github.com/pierrehoudayer/RUBIS](https://github.com/pierrehoudayer/RUBIS)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITING RUBIS -->
## Citing RUBIS

Add reference to the article

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- 
ACKNOWLEDGMENTS 
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/pierrehoudayer/RUBIS.svg?style=for-the-badge
[contributors-url]: https://github.com/pierrehoudayer/RUBIS/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/pierrehoudayer/RUBIS.svg?style=for-the-badge
[forks-url]: https://github.com/pierrehoudayer/RUBIS/network/members
[stars-shield]: https://img.shields.io/github/stars/pierrehoudayer/RUBIS.svg?style=for-the-badge
[stars-url]: https://github.com/pierrehoudayer/RUBIS/stargazers
[issues-shield]: https://img.shields.io/github/issues/pierrehoudayer/RUBIS.svg?style=for-the-badge
[issues-url]: https://github.com/pierrehoudayer/RUBIS/issues
[plot-example-1]: Plots/example_poly3_deform_at_99.99.png
[plot-example-2]: Plots/example_poly1_deform_at_99.99.png
[flowchart]: Plots/deformation_method.png
[numpy-url]: https://github.com/numpy/numpy
[scipy-url]: https://github.com/scipy/scipy
[matplotlib-url]: https://github.com/matplotlib/matplotlib
[ORCID-shield]: https://img.shields.io/badge/ORCID-0000--0002--1245--9148-brightgreen?style=flat
[ORCID-url]: https://orcid.org/0000-0002-1245-9148
