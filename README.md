# DeltaML4CCSD: Delta-Machine-Learning Models for Donor–Acceptor Complexes

This repository contains datasets, scripts, trained models, and candidate-selection workflows for delta-machine-learning prediction of CCSD energies in donor–acceptor complexes.

The central idea is to learn the correction between a lower-level quantum-chemical method and the CCSD reference energy:

```text
Delta = E_CCSD - E_low-level

The final predicted CCSD energy is then reconstructed as:

E_CCSD(predicted) = E_low-level + Delta(predicted)

The repository includes workflows based on both handcrafted molecular descriptors and SOAP descriptors.

deltaml4ccsd/
├── datasets/
│   ├── Handcrafted_Descriptors/
│   │   ├── BPBrBr/
│   │   │   ├── CCSD_MP2/
│   │   │   └── CCSD_DFT/
│   │   ├── BPClCl/
│   │   │   ├── CCSD_MP2/
│   │   │   └── CCSD_DFT/
│   │   └── BPFF/
│   │       ├── CCSD_MP2/
│   │       └── CCSD_DFT/
│   │
│   └── SOAP_Descriptors/
│       ├── BPBrBr/
│       │   ├── CCSD_MP2/
│       │   └── CCSD_DFT/
│       ├── BPClCl/
│       │   ├── CCSD_MP2/
│       │   └── CCSD_DFT/
│       └── BPFF/
│           ├── CCSD_MP2/
│           └── CCSD_DFT/
│
├── src/
└── README.md
