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

`RUBIS` (standing for *Rotation code Using Barotropy conservation over Isopotential Surfaces*) is a centrifugal deformation program that takes as input a 1D model (with spherical symmetry) and returns its deformed version by applying a conservative rotation profile specified by the user. 
More specifically, the code only needs the density as a function of radial distance, $\rho(r)$, from the reference model in addition to the surface pressure to be imposed, $P_0$, in order to perform the deformation. 
This lightness is made possible by the central procedure assumption which consists in preserving the relation between density and pressure when going from the 1D to the 2D structure. 
The latter makes it possible, in particular, to avoid the standard complications arising from the conservation of energy in the resulting model ([Jackson (1970)](https://ui.adsabs.harvard.edu/abs/1970ApJ...161..579J), [Roxburgh (2004)](https://doi.org/10.1051/0004-6361:20041202), [Jackson et al. (2005)](https://iopscience.iop.org/article/10.1086/426587), [MacDregor et al. (2007)](https://iopscience.iop.org/article/10.1086/518303)). 
In this sense, the method is analogous to the one presented by [Roxburgh et al. (2006)](https://doi.org/10.1051/0004-6361:20065109), but simpler since it does not require the calculation of the first adiabatic exponent, $\Gamma_1$, during the deformation and thus the explicit specification of an equation of state. 

As a result, the only equation effectively solved by the program is Poisson's equation, $\Delta \Phi = 4\pi G \rho$, leading to a very fast deformation of the model, even when high angular accuracies are required. 
Another feature of the method is its excellent stability, which allows for the deformation of models at speeds very close to the critical rotation rate (cf. figures [1][plot-example-1] & [2][plot-example-2] below). 
Finally, the code has been designed to allow both stellar and planetary models to be deformed, thereby dealing with potential discontinuities in the density profile. 
This is made possible by solving Poisson's equation in spheroidal rather than spherical coordinates whenever a discontinuity is present. 
More details regarding the <a href="#deformation-method">Deformation Method</a> can be found below.

Regarding the program itself, `RUBIS` is fully witten in `Python` since `v0.1.0`.

| ![Example 1][plot-example-1] | 
|:--:| 
| Deformation of a polytropic structure with index $N=3$ at $99.99$% of the critical rotation rate. Isopotentials are shown on the left and the density distribution on the right |
  
| ![Example 2][plot-example-2] | 
|:--:| 
| Idem for a $N=1$ polytrope.  |


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

As described in the flowchart, `RUBIS` uses an iterative approach to determine the deformation induced by the rotation profile. 
The method's central assumption is the preservation of the barotropic relation $\rho(P)$ over the isopotentials (surfaces that preserve the total potential, $\Phi_\mathrm{eff}$, denoted by the value of $\zeta$) which allows to have in any iteration the density profile over these surfaces, $\rho(\zeta)$. 
Depending on whether the model contains density discontinuities, the procedure takes two distinct paths:
* If the model **do not contain** any discontinuity, the density profile is first interpolated onto spherical coordinates in order to then solve Poisson's equation in this coordinate system and obtain the gravitational potential $\Phi_G(r, \theta)$. 
Because the decomposition of Poisson's equation over spherical harmonics can be decoupled, this path is the fastest one.
* If the model **do contain** discontinuities, Poisson's equation is directly solved in terms of $(\zeta, \theta)$, thus yielding $\Phi_G(\zeta, \theta)$. 
The reason for this change is that discontinuities follows isopotentials (which are also isobars from the hydrostatic equilibrium), and therefore that fixed values of $r$ cross multiple domains, making unhandly to solve the equation in the (simpler) spherical coordinate system. 
Since the isopotential shapes $\zeta(r, \theta)$ are known from the previous iteration, the gravitational can simply be reexpressed as $\Phi_G(r, \theta)$, leading to the same quantity as the other path.

Once the gravitational potential has been calculated, the total potential, $\Phi_\mathrm{eff}(r, \theta)$, is determined by adding the centrifugal potential, $\Phi_c(r, \theta)$. 
The latter is computed from the rotation profile specified by the user. 
It can be `solid`, `lorentzian` or have a `plateau` in the central region for instance, the only constraint is that it is conservative, i.e. a function of the distance from the axis of rotation, $s = r\sin\theta$, only. 
Another option for the user is to give as an input the numerical rotation profile he wants, $\Omega(s)$, and the routine will determine the appropriate centrifugal potential to use. 

In parallel, hydrostatic equilibrium implies that the total potential (expressed on the isopotentials, $\Phi_\mathrm{eff}(\zeta)$ only varies by an additive constant from one iteration to the next, a constant that is determined by applying this relationship to the central point. 
Since $\Phi_\mathrm{eff}(r, \theta)$ and $\Phi_\mathrm{eff}(\zeta)$ must correspond at the same physical locations, the program determines a new shape for the isopotentials by solving the equation $\Phi_\mathrm{eff}(\zeta) = \Phi_\mathrm{eff}(r, \theta)$. 
Once $\zeta(r, \theta)$ have been found, the matter is simply redistributed over these isopotentials and the program is ready to perform a new iteration.

The code runs until it meets a convergence criterion, typically if the polar radius of the deformed model changes less than a user-specified threshold from one iteration to the next.

| ![Method][flowchart] | 
|:--:| 
| Flowchart illustrating how the model deformation method works. Each step shows the quantity that is obtained at the end of the procedure and in terms of which variable it is obtained. |


On a practical level, the core of the program can be found in files `model_deform.py` and `model_deform_sph.py` (depending on whether Poisson's equation is solved in radial or spheroidal coordinates), which are the only ones the user needs to access. 
The file `rotation_profiles.py` contains the implementation of the rotation profiles (`solid`, `lorentzian`, `plateau` or `la_bidouille` in case of user-specified numerical profile), as well as the routines computing the centrifugal potential for each profiles. 
`generate_polytrope.py` contains the function used to generate 1D polytropes to be deformed and `numerical_routines.py` includes all the lower level functions used in the main programs.


<p align="right">(<a href="#readme-top">back to top</a>)</p>








<!-- USAGE EXAMPLES -->
## Usage

With the method described, we will now give a brief example of how the code can be used. 
Let's open the file `model_deform.py`. 
Most of the actions to be performed are limited to modifying the `set_params()` function. 
Let's start by choosing a 1D model to deform. 
If I do not have a model at hand (and I do not want to deform a model already present in the `./Models/` directory), I can choose to deform a polytrope of index `index` by providing a dictionary instead of a file name in `model_choice`:
```py
model_choice = DotDict(index=3.0)
```

I can specify more options to build the polytrope like its mass or radius for instance. The above dictionary is equivalent in practice to the following choice:
```py
model_choice = DotDict(
  index=3.0, surface_pressure=0.0, radius=1.0, mass=1.0, res=1001
)
``` 
with `res` indicating the number of points in the radial direction.


<p align="right">(<a href="#readme-top">back to top</a>)</p>









<!-- ROADMAP -->
## Roadmap

- [ ] Allow the user to choose a multiple index $(N_1, N_2, \ldots, N_k)$ polytrope (with potential density discontinuities on the interfaces) as a 1D model to deform.

See the [open issues](https://github.com/pierrehoudayer/RUBIS/issues) for a list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>









<!-- CONTRIBUTING -->
## Contributing

`RUBIS` is in stage of early development. 
If you have a suggestion to improve this repository, please fork the repo and create a pull request. 
You can also simply open an issue with the tag "enhancement". 
Any contribution is welcome and **greatly appreciated**!

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
