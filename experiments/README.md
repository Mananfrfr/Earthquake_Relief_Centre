# Earlier experiments

Run the main application from the repository root with `streamlit run app.py`.

These scripts have been retained for reference but are **not** part of the current application:

| File | Purpose |
| --- | --- |
| `early_streamlit_prototype.py` | Older Streamlit flow with simplified resource allocation |
| `multi_objective_allocation.py` | Standalone NSGA-II, MOEA/D and SPEA2 comparison |
| `pso_location_visualizer.py` | PSO demonstration with Matplotlib animation |
| `particle_animation_demo.py` | Interactive Matplotlib scatter animation; this is not a Haversine unit test |

To try them, install `requirements-experiments.txt` and run from the repository root, e.g. `python experiments/pso_location_visualizer.py` or `streamlit run experiments/early_streamlit_prototype.py`. They may require an interactive display and network access. The optional OpenRouteService key is loaded from the `OPENROUTESERVICE_API_KEY` environment variable.
