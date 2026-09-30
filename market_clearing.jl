# =============================================================================
# If you use this code in your research, please cite:
#=bibtex:
@article{Saifutdinov2026Wasserstein,
  author  = {Saifutdinov, Timur and Alahyari, Arman and Patsios, Charalampos and
             Titarenko, Roman and Deakin, Matthew and Liu, Qixin},
  title   = {A {W}asserstein Ball Formulation for Scalable Data-Driven
             Electricity Market Enabling Renewables, Storage, and Active
             Network Technologies},
  journal = {Journal of Modern Power Systems and Clean Energy (Under review)},
  year    = {2026}
}
=#
# =============================================================================
# A Wasserstein Ball Formulation for Scalable Data-Driven Electricity Market
# Enabling Renewables, Storage, and Active Network Technologies
#
# Companion implementation for the manuscript:
#   T. Saifutdinov, A. Alahyari, C. Patsios, R. Titarenko, M. Deakin, and Q. Liu,
#   "A Wasserstein Ball Formulation for Scalable Data-Driven Electricity Market
#   Enabling Renewables, Storage, and Active Network Technologies,"
#   Journal of Modern Power Systems and Clean Energy, 2026.
#
# Copyright (c) 2026 Timur Saifutdinov
# Licensed under the MIT License (see LICENSE file for details).
#
# Repository: https://github.com/tsaifutdinov13/Data-driven-LEM-enabling-RES-EES-and-ANTs
#
# -----------------------------------------------------------------------------
# DESCRIPTION
# -----------------------------------------------------------------------------
# This script simulates a year-long (365 days) day-ahead Local Electricity
# Market (LEM) clearing on the IEEE 119-bus distribution test feeder. Each
# market clearing covers a 24-hour horizon with 96 settlement periods
# (15-minute resolution).
#
# The framework integrates:
#   (i)   A data-driven Distributionally Robust Individual Chance Constrained
#         (DRICC) offer selection for stochastic RES suppliers, using a
#         Wasserstein ambiguity set and pre-enumeration of binary variables
#         (Proposition 1 in the paper) to obtain an exact LP.
#   (ii)  A trans-temporal EES participation and scheduling scheme with a
#         single four-parameter offer per day and endogenous charge/discharge
#         scheduling.
#   (iii) Network- and market-aware coordination of Active Network Technologies
#         (HT, SOP, D-STATCOM) embedded within the market clearing.
#
# The implementation uses JuMP with the Ipopt optimizer (open-source, NLP).
# =============================================================================

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------
using JuMP
using Ipopt
using DelimitedFiles
using Distributions
using JLD2

# =============================================================================
# SECTION 1: SIMULATION PARAMETERS
# =============================================================================

# Number of settlement periods per day (96 × 15 min = 24 hours)
T = 96;

# Duration of each settlement period in hours
Δt = 24 / T;

# Tangent of the power factor angle (used for reactive power approximation)
tan = 0.33;

# =============================================================================
# SECTION 2: INPUT DATA LOADING
# =============================================================================

# Wholesale electricity prices (system marginal price), 15-min resolution
WSP = readdlm("Config\\WS_Prices_15min.txt")

# Wind generation profile (normalized), 15-min resolution
WTR = readdlm("Config\\WT_15min.txt")

# PV generation profile (normalized), 15-min resolution
PVR = readdlm("Config\\PV_15min.txt")

# PV forecast error time series, 15-min resolution
PVE = readdlm("Config\\PV_forecast_error.txt")

# Wind forecast error time series, 15-min resolution
WTE = readdlm("Config\\WT_forecast_error.txt")

# Network branch data: from, to, resistance, reactance
Br = readdlm("Config\\Branches_118.txt")
B  = Int(maximum(Br[:, 1:2]))       # Number of buses
BR = size(Br, 1)                    # Number of branches
Br[:, 3:4] = Br[:, 3:4] / 121       # Convert to per-unit on new base

# Peak demand per bus: active power, reactive power
PD = readdlm("Config\\Peak_dem_118.txt")
De = 1 * readdlm("Config\\Demand_15min.txt")    # Demand scaling profile
Pd = PD[:, 1] ./ 1000                # Active power demand (MW)
Qd = PD[:, 2] ./ 1000                # Reactive power demand (MVAr)

# =============================================================================
# SECTION 3: ACTIVE NETWORK TECHNOLOGIES (ANTs) CONFIGURATION
# =============================================================================

# --- Hybrid Transformer (HT) at the Grid Supply Point (node 0) ---
# Reactive power compensation limit (MVAr)
Qht = 0.5

# --- Soft Open Points (SOPs) ---
# Columns: from bus, to bus, rating (MVA)
Sp = [54 62 0.2;
      77 99 0.2]
SP = size(Sp, 1)    # Number of SOPs

# --- Distribution Static Compensators (D-STATCOMs) ---
# Columns: bus, rating (MVA)
St = [35  0.2;
      111 0.2]
ST = size(St, 1)    # Number of D-STATCOMs

# =============================================================================
# SECTION 4: MARKET PARTICIPANT CONFIGURATION
# =============================================================================

# --- 4.1 Regular (dispatchable) suppliers ---
# Columns: bus, capacity (MW), fuel cost (£/MWh), electrical efficiency
Fs = [12   0.1   100   0.5;   # Diesel ICE
      34   0.05  50    0.35;  # Microturbine (gas)
      48   0.05  120   0.7;   # Fuel cell (hydrogen)
      101  0.15  75    0.55;  # Diesel ICE
      112  0.1   80    0.45]  # Diesel ICE
FS = size(Fs, 1)    # Number of regular suppliers

# --- 4.2 RES suppliers ---
# Columns: bus, capacity (MW), offer price (£/MWh), type (1=PV, 2=Wind), NRMSE
Rs = readdlm("Config\\RES_suppliers_45.txt")
RS = size(Rs, 1)    # Number of RES suppliers

# --- 4.3 EES suppliers ---
# EES cycle life model parameters: type, a, b, replacement cost (£/MWh), efficiency
ECL = [1   1.0e3  -1.14  150e3  0.9;
       2   1.5e3  -1.10  200e3  0.92;
       3   2.0e3  -1.05  250e3  0.94;
       4   2.5e3  -1.02  300e3  0.95]

# EES suppliers: bus, energy capacity (MWh), power capacity (MW), battery type
# (Uncomment the single line below to simulate a case with no storage)
# Es = [118 0.0001 0.0001 1]
Es = [5    0.2   0.1   2;
      10   0.4   0.2   4;
      17   0.1   0.05  1;
      22   0.8   0.4   2;
      23   0.3   0.1   3;
      25   0.2   0.05  2;
      31   0.3   0.15  2;
      33   0.1   0.1   2;
      46   0.3   0.1   3;
      54   0.2   0.1   1;
      62   0.6   0.3   1;
      69   0.4   0.1   1;
      72   0.3   0.15  2;
      81   0.8   0.4   2;
      90   0.4   0.1   3;
      95   0.2   0.1   2;
      99   0.3   0.1   3;
      107  0.2   0.1   4;
      113  0.9   0.3   1;
      116  0.4   0.2   3]
ES = size(Es, 1)    # Number of EES suppliers

# --- 4.4 Network customers / demand aggregators ---
# Distribution Use of System (DUoS) charges, 24-hour profile (p/kWh)
DUoS = [12.7 12.7 12.7 12.7 12.7 12.7 12.7 12.7 20 20 20 20 20 20 20 20 84.2 84.2 84.2 84.2 84.2 84.2 20 20]
DT = size(DUoS, 2)

# Time-of-use retail tariffs: 10 retailers × 48 half-hour slots (£/MWh)
Tariff = 10 * readdlm("Config\\Tariffs_10_48.txt")
TA = size(Tariff, 1)
TB = size(Tariff, 2)

# Resample DUoS and tariffs to 15-min resolution (96 periods)
Du = zeros(T)
Ta = zeros(TA, T)
for t = 1:DT
    Du[Int((t - 1) * T / DT + 1):Int(t * T / DT)] .= ones(Int(T / DT)) .* DUoS[t]
end
for t = 1:TB
    for ta = 1:TA
        Ta[ta, Int((t - 1) * T / TB + 1):Int(t * T / TB)] .= ones(Int(T / TB)) .* Tariff[ta, t]
    end
end

# =============================================================================
# SECTION 5: DRICC PARAMETERS AND HISTORICAL DATA
# =============================================================================

# Big-M constant for the MILP reformulation of the DRICC
M = 100

# RES non-delivery tolerance (ε in the paper): acceptable probability of
# under-delivery of accepted RES offers
e = 0.3

# Wasserstein radius (θ in the paper): confidence level of the ambiguity set
θ = 0.0005

# Simulation horizon (days)
D = 365

# Depth of the rolling historical window (number of samples per RES)
N = 100

# Array holding each RES supplier's historical delivery rates, per settlement
# period, with N initial samples and D additional samples appended during the
# simulation (one per simulated day)
OAV = ones(RS, T, D + N)

# Load pre-generated random day indices for reproducible sampling of forecast
# error scenarios:
#   HRAND - for building the initial historical window
#   BRAND - for selecting the daily forecast error realization
loaded_vars = load("Config\\rand_var.jld2")
HRAND = loaded_vars["HRAND"]
BRAND = loaded_vars["BRAND"]

# Initialise the historical delivery-rate window for each RES supplier
for t = 1:T
    for rs = 1:RS
        if Rs[rs, 4] == 1     # PV
            OAV[rs, t, 1:N] = (ones(N) + round.(Rs[rs, 5] * PVE[t, HRAND[rs, :]], digits = 4))
        elseif Rs[rs, 4] == 2 # Wind
            OAV[rs, t, 1:N] = (ones(N) + round.(Rs[rs, 5] * WTE[t, HRAND[rs, :]], digits = 4))
        end
    end
end

# -----------------------------------------------------------------------------
# PRE-ENUMERATION OF BINARY VARIABLES (Proposition 1 in the paper)
# -----------------------------------------------------------------------------
# For each RES supplier, we pre-assign the binary variable ρ_{r,n} based on
# the non-delivery tolerance ε. This eliminates the MILP structure and yields
# an exact LP reformulation (no loss of distributional robustness).
ρ = zeros(RS, N)
for rs = 1:RS
    for i = 1:N
        if i / N < e
            ρ[rs, i] = 1
        end
    end
end

# Number of network customers and retailers (equal to number of buses)
NC = B
RT = B

# =============================================================================
# SECTION 6: PRE-ALLOCATION OF RESULT ARRAYS
# =============================================================================

# Auxiliary dual variables for the DRICC
Σ   = zeros(D, RS * T, N)
ττ  = zeros(D, RS * T)

# Accepted quantities (supply side)
OFA = zeros(D, FS, T)   # Regular suppliers
ORA = zeros(D, RS * T)  # RES suppliers

# Accepted quantities (demand side)
BCA = zeros(D, NC, T)   # Network customers
BRA = zeros(D, RT, T)   # Retailers

# EES scheduling
SAC = zeros(D, ES, T)   # Charging
SAD = zeros(D, ES, T)   # Discharging
SE  = zeros(D, ES, T+1) # State of Charge

# Market outcomes
OBJ  = zeros(D)         # Daily social welfare
MCP  = zeros(D, T)      # Market Clearing Price

# Network state
U  = zeros(D, B, T)     # Squared bus voltages
P  = zeros(D, BR, T)    # Branch active power flows
Q  = zeros(D, BR, T)    # Branch reactive power flows
Pg = zeros(D, T)        # GSP active power import (≤ 0)
Qg = zeros(D, T)        # GSP reactive power import

# ANT set-points
Psf = zeros(D, SP, T)   # SOP active power (from)
Qsf = zeros(D, SP, T)   # SOP reactive power (from)
Pst = zeros(D, SP, T)   # SOP active power (to)
Qst = zeros(D, SP, T)   # SOP reactive power (to)
QHT = zeros(D, T)       # HT reactive power compensation

# Reporting arrays
HRSO = zeros(D, RS * T, 5)  # Accepted RES offers (time, bus, cap, price, r)
WLFR = zeros(D)              # Daily welfare
AOV  = zeros(RS, T, D)       # Records non-delivery events
CNTT = zeros(D)              # Daily count of RES offers

# =============================================================================
# SECTION 7: YEAR-LONG SIMULATION LOOP
# =============================================================================
for d = 1:D
    println("---------------------- Day: ", d, " -----------------------------")

    # --- 7.1 Slice and sort the historical window for the current day ---
    Oav = OAV[:, :, d:d + N - 1]
    for rs = 1:RS
        for t = 1:T
            Oav[rs, t, :] = sort(Oav[rs, t, :])
        end
    end

    # =========================================================================
    # 7.2 BUILD PARTICIPATION OFFERS FOR ALL MARKET PARTICIPANTS
    # =========================================================================

    # --- Network customers / demand aggregators (bids) ---
    NCB = zeros(T, B, 3)    # bus, capacity, price
    for t = 1:T
        NCB[t, :, 1] = [1:B;]
        for b = 1:B
            NCB[t, b, 2] = 0.25 * Pd[b, 1] * De[Int((d - 1) * T) + t, rem(b, 23) + 1]
            NCB[t, b, 3] = Ta[rem(b, 10) + 1, t] - Du[t]
        end
    end

    # --- Retailers (bids) ---
    RTB = zeros(T, B, 3)
    for t = 1:T
        RTB[t, :, 1] = [1:B;]
        for b = 1:B
            RTB[t, b, 2] = Pd[b, 1] * De[Int((d - 1) * T) + t, rem(b, 23) + 1]
            RTB[t, b, 3] = WSP[Int((d - 1) * T) + t]
        end
    end

    # --- Regular (dispatchable) suppliers (offers) ---
    # Implements Eq. (10) in the paper
    FSO = zeros(T, FS, 3)
    for t = 1:T
        FSO[t, :, 1:2] = Fs[:, 1:2]
        for i = 1:FS
            FSO[t, i, 3] = Fs[i, 3] / Fs[i, 4]   # marginal cost
        end
    end

    # --- RES suppliers (offers) ---
    # Implements Eq. (11) in the paper
    RSO = zeros(RS * T, 5)
    global CNT = 0
    for t = 1:T
        for rs = 1:RS
            if Rs[rs, 4] == 1     # PV
                forecast = Rs[rs, 2] * (1 + round.(Rs[rs, 5] * PVE[t, BRAND[rs, d]], digits = 4)) * PVR[(d - 1) * T + t]
                if forecast >= 0.001
                    global CNT += 1
                    RSO[CNT, :] = [t Rs[rs, 1] forecast Rs[rs, 3] rs]
                end
            elseif Rs[rs, 4] == 2 # Wind
                forecast = Rs[rs, 2] * (1 + round.(Rs[rs, 5] * WTE[t, BRAND[rs, d]], digits = 4)) * WTR[(d - 1) * T + t]
                if forecast >= 0.001
                    global CNT += 1
                    RSO[CNT, :] = [t Rs[rs, 1] forecast Rs[rs, 3] rs]
                end
            end
        end
    end
    RSO = RSO[1:CNT, :]

    # --- EES suppliers (single four-parameter arbitrage offers) ---
    # Implements Eq. (12) in the paper
    dod = 1
    ESO = zeros(ES, 5)          # bus, energy cap, power cap, initial SoC, price
    ESO[:, 1:3] = Es[:, 1:3]
    ESO[:, 4] = zeros(ES)
    for i = 1:ES
        # Marginal energy throughput cost (Eq. 12d)
        ESO[i, 5] = -(ECL[Int(Es[i, 4]), 3] + 1) * ECL[Int(Es[i, 4]), 4] /
                    (Es[i, 2] * ECL[Int(Es[i, 4]), 2] * ECL[Int(Es[i, 4]), 5] *
                     dod^(ECL[Int(Es[i, 4]), 3] + 2))
    end

    # =========================================================================
    # 7.3 MARKET CLEARING FORMULATION
    # =========================================================================
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
                                         "tol" => 1e-6,
                                         "max_iter" => 10000,
                                         "print_level" => 3))

    # --- Decision variables ---
    @variable(m, 0 <= ofa[fs = 1:FS, t = 1:T] <= FSO[t, fs, 2])
    @variable(m, 0 <= ora[cnt = 1:CNT] <= RSO[cnt, 3])
    @variable(m, 0 <= bca[nc = 1:NC, t = 1:T] <= NCB[t, nc, 2])
    @variable(m, 0 <= bra[rt = 1:RT, t = 1:T] <= RTB[t, rt, 2])
    @variable(m, 0 <= sac[es = 1:ES, t = 1:T] <= ESO[es, 3])
    @variable(m, 0 <= sad[es = 1:ES, t = 1:T] <= ESO[es, 3])
    @variable(m, 0 <= se[es = 1:ES, t = 1:T+1] <= ESO[es, 2])
    @variable(m, 0 <= σ[cnt = 1:CNT, n = 1:N])
    @variable(m, τ[cnt = 1:CNT])

    # Network variables (LinDistFlow)
    @variable(m, 0.94^2 <= u[b = 1:B, t = 1:T] <= 1.1^2)
    @variable(m, p[br = 1:BR, t = 1:T])
    @variable(m, q[br = 1:BR, t = 1:T])
    @variable(m, pg[t = 1:T] <= 0)
    @variable(m, qg[t = 1:T])

    # ANT variables
    @variable(m, -Sp[sp, 3] <= psf[sp = 1:SP, t = 1:T] <= Sp[sp, 3])
    @variable(m, -Sp[sp, 3] <= qsf[sp = 1:SP, t = 1:T] <= Sp[sp, 3])
    @variable(m, -Sp[sp, 3] <= pst[sp = 1:SP, t = 1:T] <= Sp[sp, 3])
    @variable(m, -Sp[sp, 3] <= qst[sp = 1:SP, t = 1:T] <= Sp[sp, 3])
    @variable(m, -St[st, 2] <= qsc[st = 1:ST, t = 1:T] <= St[st, 2])
    @variable(m, -Qht <= qht[t = 1:T] <= Qht)

    # --- Constraints ---

    # (1b) Multi-period trading balance (relaxed by EES scheduling)
    @constraint(m, con1[t = 1:T],
        sum(ofa[fs, t] for fs = 1:FS) +
        sum(ora[cnt] for cnt = 1:CNT if RSO[cnt, 1] == t) +
        sum(sad[es, t] for es = 1:ES) ==
        sum(bca[nc, t] for nc = 1:NC) +
        sum(bra[rt, t] for rt = 1:RT) +
        sum(sac[es, t] for es = 1:ES))

    # (5) DRICC reformulation of RES offer selection
    @constraint(m, con2[cnt = 1:CNT],
        e * N * τ[cnt] - sum(σ[cnt, n] for n = 1:N) >= θ * N * RSO[cnt, 3])
    @constraint(m, con3[cnt = 1:CNT, n = 1:N],
        RSO[cnt, 3] * Oav[Int(RSO[cnt, 5]), Int(RSO[cnt, 1]), n] - ora[cnt] +
        M * ρ[Int(RSO[cnt, 5]), n] >= τ[cnt] - σ[cnt, n])
    @constraint(m, con4[cnt = 1:CNT, n = 1:N],
        M * (1 - ρ[Int(RSO[cnt, 5]), n]) >= τ[cnt] - σ[cnt, n])

    # (1c)–(1d) EES energy continuity
    @constraint(m, con5[es = 1:ES], se[es, 1] == ESO[es, 2] * ESO[es, 4])
    @constraint(m, con6[es = 1:ES, t = 1:T],
        se[es, t+1] == se[es, t] + sac[es, t] * Δt - sad[es, t] * Δt)
    @constraint(m, con7[es = 1:ES], se[es, 1] == se[es, T+1])

    # (8a) LinDistFlow voltage drop
    @constraint(m, con8[br = 1:BR, t = 1:T],
        u[Int(Br[br, 2]), t] == u[Int(Br[br, 1]), t] -
        2 * (Br[br, 3] * p[br, t] + Br[br, 4] * q[br, t]))

    # (8b) Active power balance at each bus
    @constraint(m, con9a[t = 1:T],
        - sum(ofa[fs, t] for fs = 1:FS if FSO[t, fs, 1] == 1) -
        sum(ora[cnt] for cnt = 1:CNT if (RSO[cnt, 1] == t) && (RSO[cnt, 2] == 1)) +
        sum(bca[nc, t] for nc = 1:NC if NCB[t, nc, 1] == 1) -
        sum(sad[es, t] for es = 1:ES if ESO[es, 1] == 1) +
        sum(sac[es, t] for es = 1:ES if ESO[es, 1] == 1) -
        sum(p[br, t] for br = 1:BR if Br[br, 2] == 1) +
        sum(p[br, t] for br = 1:BR if Br[br, 1] == 1) +
        Pd[1] * De[Int((d - 1) * T) + t, rem(1, 23) + 1] + pg[t] +
        sum(psf[sp, t] for sp = 1:SP if 1 == Sp[sp, 1]) +
        sum(pst[sp, t] for sp = 1:SP if 1 == Sp[sp, 2]) == 0)

    @constraint(m, con9b[b = 2:B, t = 1:T],
        - sum(ofa[fs, t] for fs = 1:FS if FSO[t, fs, 1] == b) -
        sum(ora[cnt] for cnt = 1:CNT if (RSO[cnt, 1] == t) && (RSO[cnt, 2] == b)) +
        sum(bca[nc, t] for nc = 1:NC if NCB[t, nc, 1] == b) -
        sum(sad[es, t] for es = 1:ES if ESO[es, 1] == b) +
        sum(sac[es, t] for es = 1:ES if ESO[es, 1] == b) -
        sum(p[br, t] for br = 1:BR if Br[br, 2] == b) +
        sum(p[br, t] for br = 1:BR if Br[br, 1] == b) +
        Pd[b] * De[Int((d - 1) * T) + t, rem(b, 23) + 1] +
        sum(psf[sp, t] for sp = 1:SP if b == Sp[sp, 1]) +
        sum(pst[sp, t] for sp = 1:SP if b == Sp[sp, 2]) == 0)

    # (8c) Reactive power balance at each bus
    @constraint(m, con10a[t = 1:T],
        qht[t] + tan * (- sum(ofa[fs, t] for fs = 1:FS if FSO[t, fs, 1] == 1) -
        sum(ora[cnt] for cnt = 1:CNT if (RSO[cnt, 1] == t) && (RSO[cnt, 2] == 1)) +
        sum(bca[nc, t] for nc = 1:NC if NCB[t, nc, 1] == 1)) -
        sum(q[br, t] for br = 1:BR if Br[br, 2] == 1) +
        sum(q[br, t] for br = 1:BR if Br[br, 1] == 1) +
        Qd[1] * De[Int((d - 1) * T) + t, rem(1, 23) + 1] + qg[t] +
        sum(qsf[sp, t] for sp = 1:SP if 1 == Sp[sp, 1]) +
        sum(qst[sp, t] for sp = 1:SP if 1 == Sp[sp, 2]) == 0)

    @constraint(m, con10b[b = 2:B, t = 1:T],
        sum(qsc[st, t] for st = 1:ST if St[st, 1] == b) +
        tan * (- sum(ofa[fs, t] for fs = 1:FS if FSO[t, fs, 1] == b) -
        sum(ora[cnt] for cnt = 1:CNT if (RSO[cnt, 1] == t) && (RSO[cnt, 2] == b)) +
        sum(bca[nc, t] for nc = 1:NC if NCB[t, nc, 1] == b)) -
        sum(q[br, t] for br = 1:BR if Br[br, 2] == b) +
        sum(q[br, t] for br = 1:BR if Br[br, 1] == b) +
        Qd[b] * De[Int((d - 1) * T) + t, rem(b, 23) + 1] +
        sum(qsf[sp, t] for sp = 1:SP if b == Sp[sp, 1]) +
        sum(qst[sp, t] for sp = 1:SP if b == Sp[sp, 2]) == 0)

    # (8d) SOP active power balance
    @constraint(m, con11[sp = 1:SP, t = 1:T], psf[sp, t] == -pst[sp, t])

    # --- Objective function (Eq. 1a) ---
    @objective(m, Max,
        - sum(ora[cnt] * RSO[cnt, 4] for cnt = 1:CNT) +
        sum(sum(bca[nc, t] * NCB[t, nc, 3] for nc = 1:NC) +
            sum(bra[rt, t] * RTB[t, rt, 3] for rt = 1:RT) -
            sum(ofa[fs, t] * FSO[t, fs, 3] for fs = 1:FS) -
            sum((sac[es, t] + sad[es, t]) * ESO[es, 5] / 2 for es = 1:ES)
            for t = 1:T))

    # =========================================================================
    # 7.4 SOLVE AND STORE RESULTS
    # =========================================================================
    optimize!(m)

    Σ[d, 1:CNT, :] = JuMP.value.(σ)
    ττ[d, 1:CNT]    = JuMP.value.(τ)
    OFA[d, :, :]    = JuMP.value.(ofa)
    BCA[d, :, :]    = JuMP.value.(bca)
    ORA[d, 1:CNT]   = JuMP.value.(ora)
    BRA[d, :, :]    = JuMP.value.(bra)
    SAC[d, :, :]    = JuMP.value.(sac)
    SAD[d, :, :]    = JuMP.value.(sad)
    SE[d, :, :]     = JuMP.value.(se)
    OBJ[d]          = JuMP.objective_value(m)
    MCP[d, :]       = JuMP.dual.(con1)
    U[d, :, :]      = JuMP.value.(u)
    P[d, :, :]      = JuMP.value.(p)
    Q[d, :, :]      = JuMP.value.(q)
    Pg[d, :]        = JuMP.value.(pg)
    Qg[d, :]        = JuMP.value.(qg)
    Psf[d, :, :]    = JuMP.value.(psf)
    Qsf[d, :, :]    = JuMP.value.(qsf)
    Pst[d, :, :]    = JuMP.value.(pst)
    Qst[d, :, :]    = JuMP.value.(qst)
    WLFR[d]         = JuMP.objective_value(m)
    QHT[d, :]       = JuMP.value.(qht)
    HRSO[d, 1:CNT, :] = RSO

    # --- 7.5 Update rolling historical window ---
    # After the market clears and actual generation is realized, update the
    # delivery-rate history by appending the current day's realization.
    for cnt = 1:CNT
        if Rs[Int(RSO[cnt, 5]), 4] == 1     # PV
            OAV[Int(RSO[cnt, 5]), Int(RSO[cnt, 1]), N + d] =
                RSO[cnt, 3] / (Rs[Int(RSO[cnt, 5]), 2] * PVR[(d - 1) * T + Int(RSO[cnt, 1])])
        elseif Rs[Int(RSO[cnt, 5]), 4] == 2 # Wind
            OAV[Int(RSO[cnt, 5]), Int(RSO[cnt, 1]), N + d] =
                RSO[cnt, 3] / (Rs[Int(RSO[cnt, 5]), 2] * WTR[(d - 1) * T + Int(RSO[cnt, 1])])
        end
    end

    # --- 7.6 Track non-delivery events ---
    CNTT[d] = CNT
    for cnt = 1:CNT
        if Rs[Int(RSO[cnt, 5]), 4] == 1
            if ORA[d, cnt] >= Rs[Int(RSO[cnt, 5]), 2] * PVR[(d - 1) * T + Int(RSO[cnt, 1])]
                AOV[Int(RSO[cnt, 5]), Int(RSO[cnt, 1]), d] = 1
            end
        elseif Rs[Int(RSO[cnt, 5]), 4] == 2
            if ORA[d, cnt] >= Rs[Int(RSO[cnt, 5]), 2] * WTR[(d - 1) * T + Int(RSO[cnt, 1])]
                AOV[Int(RSO[cnt, 5]), Int(RSO[cnt, 1]), d] = 1
            end
        end
    end
end