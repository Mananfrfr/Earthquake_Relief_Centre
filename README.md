# Earthquake Relief Centre

A **Streamlit** research prototype for earthquake disaster-site mapping, optimized relief-centre placement and resource allocation using Jaipur as an example.

## Features

- Interactive Folium map and disaster-site data entry
- Particle Swarm Optimization (PSO) for proposed relief-centre locations
- Supply capacity input and relief-centre location review
- Multi-objective allocation using NSGA-II, MOEA/D and SPEA2
- Comparisons, plots, summaries and downloadable CSVs

## Quick start

Python 3.10+ (Python 3.11 recommended):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The current application uses **Streamlit**, not the Flask/MySQL stack described in the former README. It manages the working scenario in session state and does not have a production database.

## Workflow

1. Set the number of disaster sites and relief centres and select an optimization mode.
2. Mark sites on the map and record demand, severity and affected population.
3. Provide relief-centre supply capacities.
4. Run PSO to propose locations; review and adjust them.
5. Run the allocation algorithms and inspect the results or download a CSV.

## Code organization

```text
app.py                                 Streamlit entry point
relief_optimization/pso.py              PSO fitness and geographic helpers
scripts/export_jaipur_road_network.py   Optional road-data downloader
experiments/                             Historical algorithm/UI prototypes
tests/test_pso.py                        PSO helper unit tests
docs/architecture.md                     System overview
requirements.txt                         Dashboard dependencies
requirements-experiments.txt             Extra standalone dependencies
```

Run the main project with `streamlit run app.py`. Do not run the historical `experiments/` scripts unless you specifically want to try those standalone demonstrations.

For optional experiments:

```bash
pip install -r requirements-experiments.txt
python scripts/export_jaipur_road_network.py
```

The road export requires internet access and writes to the ignored local `data/` folder. The old standalone allocation experiment can optionally read the `OPENROUTESERVICE_API_KEY` environment variable. Never commit credentials.

## Validation

```bash
python -m compileall -q app.py relief_optimization experiments scripts tests
python -m unittest discover -s tests -v
```

## Model limitations

**Academic demonstration, not an operational dispatch tool.** The dashboard uses Haversine straight-line distances and estimates travel time with a fixed average speed. The optional OSMnx network is **not** incorporated into the Streamlit app or its delivery-time calculation. Data are entered manually and stored per session.
