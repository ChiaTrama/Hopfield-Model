# Hopfield Model Analysis

This project implements a comprehensive analysis of the **Hopfield model** for pattern recognition and image reconstruction using Python. It was developed as part of the course **Laboratory of Computational Physics A**, academic year 2024-2025.

## Authors
- Chiara Tramarin   
- Alessio Tuscano  
- Antonio Donà 
- Alberto Schiavinato  

## Project Description
Our project evaluates the performance of the Hopfield network in pattern recognition and image reconstruction. We explore its application to both synthetic 2D random patterns and the MNIST dataset, investigating various algorithm optimizations—including alternative update rules and temperature effects—to assess the network's capabilities and limitations.

## Structure
The code is organized into three subparts:
- A class that handles the complete Hopfield model workflow, excluding the grid search.
- Helper functions for pattern initialization and corruption.
- Additional utilities, including plotting functions and the grid search implementation.

These components are integrated into the main analysis notebook where all results are compiled and visualized. The code leverages parallel processing: single-instance operations use NumPy with Python 3.13, while a manual parallel implementation is employed for the grid search to accommodate different system configurations.
