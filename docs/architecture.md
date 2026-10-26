# Architecture

```text
Disaster site / supply input
           |
     Streamlit app.py
           |
  PSO centre placement -------- relief_optimization/pso.py
           |
  User reviews locations
           |
 Resource allocation (app.py)
    /       |       \
 NSGA-II  MOEA/D   SPEA2
           |
    Tables / plots / CSV
```

The dashboard holds data in Streamlit session state. There is no Flask/MySQL service in this version.

- `app.py`: entry point, UI, PSO execution and multi-objective allocation
- `relief_optimization/pso.py`: Haversine, dynamic bounding box, PSO fitness
- `experiments/`: prior prototypes preserved for study
- `scripts/`: optional OSMnx road network export; the main app does not use it
- `tests/`: lightweight PSO helper tests

The resource-allocation UI uses approximate straight-line distances and a fixed travel speed, **not** live road routing. This is an academic prototype, not a validated emergency-response system.
