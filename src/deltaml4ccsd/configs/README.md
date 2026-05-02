# Configuration files

Each JSON file links one dataset case to the centralized Python modules in
`src/deltaml4ccsd`.

Example for handcrafted descriptors:

```bash
python -m deltaml4ccsd.train_handcrafted --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
```

Example for SOAP descriptors:

```bash
python -m deltaml4ccsd.train_soap --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json
python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json
```

At the moment, these central modules are scaffold entry points.
The dataset-local scripts remain the fully executable reference workflows
until the cleaned logic is moved into `src/deltaml4ccsd/`.
