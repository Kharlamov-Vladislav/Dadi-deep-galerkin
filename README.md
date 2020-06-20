# Deep Galerkin Method for predicting demographic history of populations

*Bioinformatics Institute, 2020, spring*

# Table of contents

1. [Project goals](#sec1) </br>
2. [Methods](#sec2)</br>
3. [Result](#sec3)</br>
4. [Startup instructions](#sec4)</br>
    4.1. [Requirements](#sec4.1)</br>
    4.2. [Dependencies](#sec4.2)</br>
    4.3. [How to run](#sec4.3)</br>
5. [Gratitude](#sec5)</br>

## Project goals
<a name="sec1"></a>
The aim of the project was to introduce the method for solving differential equations using [Deep Galerkin](https://arxiv.org/pdf/1909.11544.pdf) neural networks into the dadi method for solving the diffusion equation. Since the [Diffusion Approximations for Demographic Inference [dadi]](https://github.com/niuhuifei/dadi) method simulates genetic data, namely the allele-frequency spectrum (AFS), numerically solving several diffusion equations. 

It is proposed to solve the following equation using Deep Galerkin:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=$\displaystyle\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2}\frac{x(1-x)}{2\rho(t)}  %2B  \frac{\partial u}{\partial x}S x(1-x) = 0$">  
</p>
with boundary condition: 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=$\lim_{x \to 0} u(x,t) = \theta \rho(t)$">
</p>

where *S* is selection coefficient, <img src="https://render.githubusercontent.com/render/math?math=$\rho$"> is relative population size and <img src="https://render.githubusercontent.com/render/math?math=$\theta$"> is influx of new mutations coefficient.  

The solution u(x,t) of this equation is the density of the allele-frequency spectrum. With it, it is possible to evaluate the parameters of the demographic history of populations using the maximum likelihood method.

## Methods
<a name="sec2"></a>
The implementation from the original article of [Deep Galerkin](https://arxiv.org/pdf/1909.11544.pdf) was used with some changes. The Deep Galerkin method was applied for the one-dimensional diffusion equation with the following parameters: selection, relative population size and mutation influx. We also implemented training with different sets of parameters, which allowed us to train the model once, and use it further with different parameters without re-training.  

The model that is used to approximate the solution is a neural network. It consists of one Dense layer as input, specified number of hidden LSTM layers and one Dense layer as output.  
This model is trained on randomly generated samples from PDE time-space domain <img src="https://render.githubusercontent.com/render/math?math=$%5B0,%201%5D%20%5Ctimes%20%5B0,1%5D"> and from parameters domain.  
The loss function is consists of three terms: 
  * to minimize differential equation operator: 
  
  <p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=$\displaystyle\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2}\frac{x(1-x)}{2\rho(t)}  %2B  \frac{\partial u}{\partial x}S x(1-x)$">  
 </p>
  which is calculated analytically at point (x, t, rho, S, theta) from generated data.
  
  * to match the solution to the boundary conditions  
  <p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=$u(0,t) - \theta\rho(t)$">
 </p>
 
  * to match the solution to the initial conditions  
 <p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=$u(x,0) - \rho\theta\frac{1 - exp(-2S(1-x))}{1 - exp(-2S)}$">
  </p>
    
## Results
<a name="sec3"></a>
We obtained almost identical solution with dadi method for AFS with RMSE less than 1 on average. We have not obtained the desired acceleration, because model training is slow. However it seems that for large dimensions, the method based on neural networks should give on orders of magnitude better result than the classical methods for solving differential equations, and in particular, the diffusion equation.

From the application of our method we can get the following result, for example:

![Comparison AFS of our method with the numerical solution](docs/afs_example.png)
![Comparison density of our method with the numerical solution](docs/density_example.png)

# Startup instructions
<a name="sec4"></a>
## Requirements
<a name="sec4.1"></a>
Python >= 3.7

(Tested on Ubuntu 18.04, Python 3.7)

### Dependencies
<a name="sec4.2"></a>
We recommend using a virtual environment to eliminate dependency errors.

```
pip install -r requirements.txt
```

to install dependencies.

### Running
<a name="sec4.3"></a>
Run `model_train.py` to train a new model to solve the diffusion equation. The trained model will be stored in "trained_models" folder.
There are no parameters to set in CLI. You can change the range of parameters of equation and training settings by editing the source code.  
**You can change the following parameters:**  


#### PDE parameters domain:  
  * nu_low and nu_high: range of values for a relative population size  
  * gamma_low and gamma_high: range of values for a selection coefficient  
  * theta_low and theta_high: range of values for the influx of new mutations coefficient  
  
  Large range of values for parameters require more iterations of training, but it will give you a more versatile model, that can be used with wider domain of parameters.  
  
#### Neural network architecture parameters:  
  * num_layers: the number of layers  
  * nodes_per_layer: the number of neurons in the layer    
#### Training parameters:  
  * learning_rate: the step size at each iteration of gradient descent  
  * sampling_stages: the number of times to resample new time-space domain points  
  * steps_per_sample: the number of stochastic gradient descent steps to take before re-sampling  
  * nSim_interior: size of samples from time-space domain and parameters domain  
    
You can use an already trained model that is saved in a folder ```trained_models```. The example of its use and comparison with the classical solution is in ```one_pop_example.ipynb``` notebook.

## Gratitude
<a name="sec5"></a>

