# A Wasserstein Ball Formulation for Scalable Data-Driven Electricity Market Enabling Renewables, Storage, and Active Network Technologies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-blueviolet)](https://julialang.org/)

This repository contains the complete reference implementation of the data-driven, distributionally robust Local Electricity Market (LEM) clearing framework proposed in:

> **T. Saifutdinov, A. Alahyari, C. Patsios, R. Titarenko, M. Deakin, and Q. Liu,**
> *"A Wasserstein Ball Formulation for Scalable Data-Driven Electricity Market Enabling Renewables, Storage, and Active Network Technologies,"*
> Journal of Modern Power Systems and Clean Energy (Under review), 2026.

The framework integrates three key innovations within a single computationally efficient Linear Program (LP):

1. **RES integration scheme** — a data-driven Distributionally Robust Individual Chance Constrained (DRICC) offer selection formulation, using a Wasserstein ambiguity set and pre-enumeration of binary variables (Proposition 1) to obtain an **exact** LP.
2. **EES integration scheme** — a trans-temporal participation mechanism based on a single four-parameter offer per day, combined with a market-side scheduling formulation that endogenously determines the optimal charge/discharge schedule.
3. **ANT coordination scheme** — network- and market-aware coordination of Hybrid Transformers (HT), Soft Open Points (SOP), and Distribution Static Compensators (D-STATCOM) embedded within the market clearing.

## Key Features

- **Scalable LP formulation**: clears ~20,000 bids and offers per day in a few minutes.
- **Distributional robustness**: the Wasserstein-ball DRICC preserves the robustness guarantee while eliminating the MILP structure via exact pre-enumeration.
- **Trans-temporal EES scheduling**: endogenous charge/discharge scheduling that removes bidding risks.
- **Network-aware coordination**: LinDistFlow network model with embedded ANT models.
- **Year-long simulation**: 365-day rolling-horizon market clearing on the IEEE 119-bus distribution test feeder.

## Repository Structure

├── market_clearing.jl # Main simulation script
├── Config/ # Input data directory
│ ├── WS_Prices_15min.txt # Wholesale electricity prices
│ ├── WT_15min.txt # Wind generation profile
│ ├── PV_15min.txt # PV generation profile
│ ├── PV_forecast_error.txt # PV forecast error time series
│ ├── WT_forecast_error.txt # Wind forecast error time series
│ ├── Branches_118.txt # Network branch data
│ ├── Peak_dem_118.txt # Peak demand per bus
│ ├── Demand_15min.txt # Demand scaling profile
│ ├── RES_suppliers_45.txt # RES supplier configuration
│ └── Tariffs_10_48.txt # Retail tariffs
├── rand_var.jld2 # Pre-generated random day indices
├── LICENSE # MIT License
├── CITATION.cff # Citation metadata
└── README.md # This file


## Installation

### 1. Install Julia

Download and install Julia (version 1.9 or later) from [julialang.org](https://julialang.org/downloads/).

### 2. Install dependencies

Open a Julia REPL and run:

```julia
using Pkg
Pkg.add(["JuMP", "Ipopt", "DelimitedFiles", "Distributions", "JLD2"])

##Usage

### 1. Run the main simulation script from the repository root:

include("market_clearing.jl")