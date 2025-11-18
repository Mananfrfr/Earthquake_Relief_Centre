# app.py
"""
Earthquake Relief System - Streamlit app (updated & patched)
- Requires: streamlit, folium, streamlit_folium, pymoo, plotly, numpy, pandas
- Uses PSO core at: /mnt/data/pso_core.py
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import time
import math
import random
import io
import plotly.express as px
import plotly.graph_objects as go

# Import PSO fitness & helpers from uploaded module
from pso_core import fitness_function, disaster_sites as default_disaster_sites, haversine_km

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DEFAULT_CENTER = [26.9124, 75.7873]
LAT_MIN, LAT_MAX = 26.80, 27.05
LON_MIN, LON_MAX = 75.70, 76.00

# PSO hyperparams
SWARM_SIZE = 30
MAX_ITER = 50
W = 0.7
C1 = 1.5
C2 = 1.5

# Default MOO hyperparams (full)
DEFAULT_NSGA_POP = 100
DEFAULT_NSGA_GEN = 100
DEFAULT_SPEA_POP = 100
DEFAULT_SPEA_GEN = 100
DEFAULT_MOEAD_POP = 80
DEFAULT_MOEAD_GEN = 80

# Quick-mode hyperparams (for fast dev/testing)
QUICK_NSGA_POP = 40
QUICK_NSGA_GEN = 40
QUICK_SPEA_POP = 40
QUICK_SPEA_GEN = 40
QUICK_MOEAD_POP = 40
QUICK_MOEAD_GEN = 40

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
def init_session_state():
    if 'disaster_sites' not in st.session_state:
        st.session_state.disaster_sites = []
    if 'relief_centres' not in st.session_state:
        st.session_state.relief_centres = []
    if 'num_disaster_sites' not in st.session_state:
        st.session_state.num_disaster_sites = 0
    if 'num_relief_centres' not in st.session_state:
        st.session_state.num_relief_centres = 5
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'setup'
    if 'pso_completed' not in st.session_state:
        st.session_state.pso_completed = False
    if 'moo_results' not in st.session_state:
        st.session_state.moo_results = None
    if 'last_click' not in st.session_state:
        st.session_state.last_click = None
    if 'user_satisfied' not in st.session_state:
        st.session_state.user_satisfied = None
    if 'moo_mode' not in st.session_state:
        st.session_state.moo_mode = 'Full'  # Full or Quick

# ---------------------------------------------------------------------
# PSO Implementation (kept compatible with pso_core fitness)
# ---------------------------------------------------------------------
class Particle:
    def __init__(self, num_relief_centers):
        self.dimensions = num_relief_centers * 2
        self.position = [
            random.uniform(LAT_MIN, LAT_MAX) if i % 2 == 0 else random.uniform(LON_MIN, LON_MAX)
            for i in range(self.dimensions)
        ]
        self.velocity = [random.uniform(-0.01, 0.01) for _ in range(self.dimensions)]
        self.best_position = list(self.position)
        self.best_fitness = float('inf')

def run_pso_algorithm(disaster_sites, num_relief_centers=5):
    if not disaster_sites:
        return []
    DIMENSIONS = num_relief_centers * 2
    swarm = [Particle(num_relief_centers) for _ in range(SWARM_SIZE)]
    global_best_position = None
    global_best_fitness = float('inf')

    for particle in swarm:
        fitness = fitness_function(
            particle.position,
            disaster_sites,
            weight_distance=1.0,
            weight_proximity=2.0,
            weight_spread=1.0,
            weight_coverage=2.0
        )
        particle.best_fitness = fitness
        particle.best_position = particle.position[:]
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position[:]

    progress_bar = st.progress(0)
    status_text = st.empty()
    for iteration in range(MAX_ITER):
        for particle in swarm:
            for i in range(DIMENSIONS):
                r1 = random.random()
                r2 = random.random()
                cognitive = C1 * r1 * (particle.best_position[i] - particle.position[i])
                social = C2 * r2 * (global_best_position[i] - particle.position[i])
                particle.velocity[i] = W * particle.velocity[i] + cognitive + social
                particle.position[i] += particle.velocity[i]
                # boundaries
                if i % 2 == 0:
                    particle.position[i] = max(min(particle.position[i], LAT_MAX), LAT_MIN)
                else:
                    particle.position[i] = max(min(particle.position[i], LON_MAX), LON_MIN)
            fitness = fitness_function(
                particle.position,
                disaster_sites,
                weight_distance=1.0,
                weight_proximity=2.0,
                weight_spread=1.0,
                weight_coverage=2.0
            )
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position[:]
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position[:]
        progress_bar.progress((iteration + 1) / MAX_ITER)
        status_text.text(f"Iteration {iteration+1}/{MAX_ITER} | Best Fitness: {global_best_fitness:.2f}")
    progress_bar.empty()
    status_text.empty()

    relief_centres = []
    for i in range(num_relief_centers):
        lat = global_best_position[2 * i]
        lon = global_best_position[2 * i + 1]
        nearest_site = min(
            disaster_sites,
            key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"])
        )
        relief_centres.append({
            "id": f"RC{i}",
            "name": f"Relief Centre {i+1}",
            "lat": lat,
            "lon": lon,
            "supply": random.randint(500, 2000),
            "area": f"Near {nearest_site.get('name', 'Unknown')}",
            "source": "PSO"
        })
    return relief_centres

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def haversine_distance_time(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    urban_speed_kmh = 30
    time_min = (distance / urban_speed_kmh) * 60
    return distance, time_min

def build_distance_time_matrices(relief_centres, disaster_sites):
    distance_matrix = []
    time_matrix = []
    for rc in relief_centres:
        distances = []
        times = []
        for ds in disaster_sites:
            d, t = haversine_distance_time(rc["lat"], rc["lon"], ds["lat"], ds["lon"])
            distances.append(d)
            times.append(t)
        distance_matrix.append(distances)
        time_matrix.append(times)
    return distance_matrix, time_matrix

# ---------------------------------------------------------------------
# MOO: NSGA-II (constrained), SPEA2 (constrained), MOEA/D (unconstrained)
# ---------------------------------------------------------------------
def run_moo_algorithms(relief_centres, disaster_sites, mode='Full'):
    """
    Runs NSGA-II, MOEA/D and SPEA2 using a proportional/allocation MOO formulation.
    - Constraints:
        * For each disaster site:  total_allocated_to_site <= demand_site
        * For each relief centre: total_allocated_from_rc <= supply_rc
    - Objectives:
        f1 = total_time + alpha_unused * total_unused_supply
        f2 = total_distance + alpha_unused * total_unused_supply
        f3 = -priority_fulfillment + alpha_unused * total_unused_supply
    - mode: 'Full' (default) or 'Quick' (smaller pop/gens)
    """
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.algorithms.moo.spea2 import SPEA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.decomposition.tchebicheff import Tchebicheff
        from pymoo.util.ref_dirs import get_reference_directions
    except Exception as e:
        st.error("`pymoo` is required to run MOO algorithms. Install it in your environment (pip install pymoo).")
        st.exception(e)
        return None

    # pick hyperparams
    if mode == 'Quick':
        nsga_pop, nsga_gen = QUICK_NSGA_POP, QUICK_NSGA_GEN
        spea_pop, spea_gen = QUICK_SPEA_POP, QUICK_SPEA_GEN
        moead_pop, moead_gen = QUICK_MOEAD_POP, QUICK_MOEAD_GEN
    else:
        nsga_pop, nsga_gen = DEFAULT_NSGA_POP, DEFAULT_NSGA_GEN
        spea_pop, spea_gen = DEFAULT_SPEA_POP, DEFAULT_SPEA_GEN
        moead_pop, moead_gen = DEFAULT_MOEAD_POP, DEFAULT_MOEAD_GEN

    # build priority scores for disaster sites (used in objective)
    max_people = max((ds.get("people_affected", 1) for ds in disaster_sites), default=1)
    max_sev = max((ds.get("severity_level", 1) for ds in disaster_sites), default=1)
    normalization = max_people * max_sev if max_people * max_sev > 0 else 1
    for ds in disaster_sites:
        ds["priority_score"] = (ds.get("people_affected", 1) * ds.get("severity_level", 1)) / normalization

    # demand and supply vectors
    demand_vector = np.array([ds.get("demand", 0) for ds in disaster_sites], dtype=float)
    supply_vector = np.array([rc.get("supply", 0) for rc in relief_centres], dtype=float)

    # distance/time matrices
    distance_matrix, time_matrix = build_distance_time_matrices(relief_centres, disaster_sites)

    # pre-check: warn if total supply < number of sites (still allowed)
    total_supply = float(np.sum(supply_vector))
    total_min_required = len(disaster_sites)
    if total_supply < 1:
        st.warning(f"Total available supply ({total_supply}) is extremely low — results may be degenerate.")

    # penalty weight for unused supply (user chose medium)
    ALPHA_UNUSED = 1.0

    # Helper sizes
    n_rc = len(relief_centres)
    n_ds = len(disaster_sites)
    n_var = n_rc * n_ds

    # upper bounds per variable: cannot allocate more than rc supply or ds demand
    xu = np.array([min(supply_vector[rc_idx], demand_vector[ds_idx])
                   for rc_idx in range(n_rc) for ds_idx in range(n_ds)], dtype=float)
    xl = np.zeros(n_var, dtype=float)

    # -------------------------------
    # Constrained problem (NSGA-II / SPEA2)
    # Constraints G: first n_ds entries -> sum_alloc_to_ds - demand_ds <= 0
    #                next n_rc entries  -> sum_alloc_from_rc - supply_rc <= 0
    # -------------------------------
    class ResourceAllocationProblem(Problem):
        def __init__(self, relief_centers, disaster_sites, distance_matrix, time_matrix):
            self.relief_centers = relief_centers
            self.disaster_sites = disaster_sites
            self.distance_matrix = distance_matrix
            self.time_matrix = time_matrix
            n_var_local = len(relief_centers) * len(disaster_sites)
            xl_local = np.zeros(n_var_local)
            xu_local = np.array([min(rc.get("supply", 1000), ds.get("demand", 1000)) for rc in relief_centers for ds in disaster_sites])
            n_obj_local = 3
            # constraints: n_ds (demand upper bound) + n_rc (supply upper bound)
            n_constr_local = len(disaster_sites) + len(relief_centers)
            Problem.__init__(self, n_var=n_var_local, n_obj=n_obj_local, n_constr=n_constr_local, xl=xl_local, xu=xu_local)

        def _evaluate(self, x, out, *args, **kwargs):
            n_solutions = x.shape[0]
            f1 = np.zeros(n_solutions)
            f2 = np.zeros(n_solutions)
            f3 = np.zeros(n_solutions)
            G = np.zeros((n_solutions, len(self.disaster_sites) + len(self.relief_centers)))
            for i in range(n_solutions):
                # round to integers for allocation interpretation
                alloc = np.rint(x[i].reshape(len(self.relief_centers), len(self.disaster_sites))).astype(int)
                total_time = 0.0
                total_distance = 0.0
                priority_fulfillment = 0.0

                # per-DS and per-RC sums
                sum_per_ds = alloc.sum(axis=0)   # length n_ds
                sum_per_rc = alloc.sum(axis=1)   # length n_rc

                # compute objectives (time/distance/priority)
                for rc_idx in range(len(self.relief_centers)):
                    for ds_idx in range(len(self.disaster_sites)):
                        allocated = alloc[rc_idx, ds_idx]
                        if allocated > 0:
                            total_time += self.time_matrix[rc_idx][ds_idx] * allocated
                            total_distance += self.distance_matrix[rc_idx][ds_idx] * allocated
                            priority_fulfillment += allocated * self.disaster_sites[ds_idx].get("priority_score", 0)

                # unused supply penalty
                unused_supply = float(np.sum(np.maximum(supply_vector - sum_per_rc, 0.0)))
                penalty = ALPHA_UNUSED * unused_supply

                f1[i] = total_time + penalty
                f2[i] = total_distance + penalty
                f3[i] = -priority_fulfillment + penalty

                # Constraints: sum_per_ds <= demand_vector  -> sum_per_ds - demand <= 0
                G[i, 0:len(self.disaster_sites)] = sum_per_ds - demand_vector
                # Constraints: sum_per_rc <= supply_vector -> sum_per_rc - supply <= 0
                G[i, len(self.disaster_sites):] = sum_per_rc - supply_vector

            out["F"] = np.column_stack([f1, f2, f3])
            out["G"] = G

    # -------------------------------
    # Unconstrained problem for MOEA/D (we'll add soft unused-supply penalty inside objectives)
    # -------------------------------
    class UnconstrainedProblem(Problem):
        def __init__(self, relief_centers, disaster_sites, distance_matrix, time_matrix):
            self.relief_centers = relief_centers
            self.disaster_sites = disaster_sites
            self.distance_matrix = distance_matrix
            self.time_matrix = time_matrix
            n_var_local = len(relief_centers) * len(disaster_sites)
            xl_local = np.zeros(n_var_local)
            xu_local = np.array([min(rc.get("supply", 1000), ds.get("demand", 1000)) for rc in relief_centers for ds in disaster_sites])
            Problem.__init__(self, n_var=n_var_local, n_obj=3, n_constr=0, xl=xl_local, xu=xu_local)

        def _evaluate(self, x, out, *args, **kwargs):
            n_solutions = x.shape[0]
            f1 = np.zeros(n_solutions)
            f2 = np.zeros(n_solutions)
            f3 = np.zeros(n_solutions)
            for i in range(n_solutions):
                alloc = np.rint(x[i].reshape(len(self.relief_centers), len(self.disaster_sites))).astype(int)
                total_time = 0.0
                total_distance = 0.0
                priority_fulfillment = 0.0
                sum_per_ds = alloc.sum(axis=0)
                sum_per_rc = alloc.sum(axis=1)

                for rc_idx in range(len(self.relief_centers)):
                    for ds_idx in range(len(self.disaster_sites)):
                        allocated = alloc[rc_idx, ds_idx]
                        if allocated > 0:
                            total_time += self.time_matrix[rc_idx][ds_idx] * allocated
                            total_distance += self.distance_matrix[rc_idx][ds_idx] * allocated
                            priority_fulfillment += allocated * self.disaster_sites[ds_idx].get("priority_score", 0)

                # penalize allocations that exceed demand or supply (softly), and penalize unused supply
                exceed_demand = float(np.sum(np.maximum(sum_per_ds - demand_vector, 0.0)))
                exceed_supply = float(np.sum(np.maximum(sum_per_rc - supply_vector, 0.0)))
                unused_supply = float(np.sum(np.maximum(supply_vector - sum_per_rc, 0.0)))

                soft_penalty = 1e6 * (exceed_demand + exceed_supply)  # large penalty for hard violations
                leftover_penalty = ALPHA_UNUSED * unused_supply

                f1[i] = total_time + leftover_penalty + soft_penalty
                f2[i] = total_distance + leftover_penalty + soft_penalty
                f3[i] = -priority_fulfillment + leftover_penalty + soft_penalty

            out["F"] = np.column_stack([f1, f2, f3])

    problem_constrained = ResourceAllocationProblem(relief_centres, disaster_sites, distance_matrix, time_matrix)
    problem_unconstrained = UnconstrainedProblem(relief_centres, disaster_sites, distance_matrix, time_matrix)

    results = {}

    def _run_algorithm_on_problem(algorithm, problem, algo_name, termination):
        t0 = time.time()
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        t1 = time.time()
        exec_time = t1 - t0
        F = res.F if hasattr(res, "F") else np.zeros((0, 3))
        X = res.X if hasattr(res, "X") else np.zeros((0, problem.n_var))
        pareto_size = len(F)

        # Safe selection: ignore trivial all-zero solutions and any that violate hard bounds (for unconstrained)
        min_time_idx = None
        min_time_solution = None
        if F.shape[0] > 0:
            # mask out trivial solutions (all objectives nearly zero)
            non_trivial_mask = ~np.all(np.isclose(F, 0.0, atol=1e-6), axis=1)
            if np.any(non_trivial_mask):
                valid_idxs = np.where(non_trivial_mask)[0]
                # among valid, choose min time
                rel_idx = int(np.argmin(F[valid_idxs, 0]))
                min_time_idx = int(valid_idxs[rel_idx])
                min_time_solution = F[min_time_idx]
            else:
                # if all trivial, keep None
                min_time_idx = None
                min_time_solution = None

        allocation = None
        if min_time_idx is not None and X.shape[0] > 0:
            allocation = {}
            vec = X[min_time_idx]
            alloc_mat = np.rint(vec.reshape(len(relief_centres), len(disaster_sites))).astype(int)
            for rc_idx, rc in enumerate(relief_centres):
                for ds_idx, ds in enumerate(disaster_sites):
                    allocated = int(alloc_mat[rc_idx, ds_idx])
                    if allocated > 0:
                        allocation[(rc.get("id", rc.get("name", f"RC{rc_idx}")),
                                    ds.get("id", ds.get("name", f"DS{ds_idx}")))] = allocated

        return {
            'res_obj': res,
            'exec_time': exec_time,
            'pareto_size': pareto_size,
            'min_time_idx': min_time_idx,
            'min_time_solution': min_time_solution,
            'allocation': allocation,
            'F': F,
            'X': X
        }

    # NSGA-II (constrained)
    nsga_algo = NSGA2(
        pop_size=nsga_pop,
        n_offsprings=nsga_pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    st.info("Running NSGA-II (constrained)")
    nsga_res = _run_algorithm_on_problem(nsga_algo, problem_constrained, "NSGA-II", ('n_gen', nsga_gen))
    results['NSGA-II'] = nsga_res

    # MOEA/D (unconstrained but with penalties)
    st.info("Running MOEA/D (unconstrained)")
    # choose reference directions (tune partitions if needed)
    ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)
    moead_algo = MOEAD(
        ref_dirs=ref_dirs,
        decomposition=Tchebicheff(),
        n_neighbors=15,
        prob_neighbor_mating=0.7,
        seed=42,
        verbose=False
    )
    moead_res = _run_algorithm_on_problem(moead_algo, problem_unconstrained, "MOEAD", ('n_gen', moead_gen))
    results['MOEAD'] = moead_res

    # SPEA2 (constrained)
    st.info("Running SPEA2 (constrained)")
    spea2_algo = SPEA2(
        pop_size=spea_pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    spea2_res = _run_algorithm_on_problem(spea2_algo, problem_constrained, "SPEA2", ('n_gen', spea_gen))
    results['SPEA2'] = spea2_res

    # Build performance DataFrame (safe min fetch)
    def _safe_min_time(sol):
        return sol[0] if sol is not None else float('inf')

    perf = pd.DataFrame({
        'Algorithm': ['NSGA-II', 'MOEAD', 'SPEA2'],
        'Execution Time (s)': [results['NSGA-II']['exec_time'], results['MOEAD']['exec_time'], results['SPEA2']['exec_time']],
        'Pareto Front Size': [results['NSGA-II']['pareto_size'], results['MOEAD']['pareto_size'], results['SPEA2']['pareto_size']],
        'Min Delivery Time': [
            _safe_min_time(results['NSGA-II']['min_time_solution']),
            _safe_min_time(results['MOEAD']['min_time_solution']),
            _safe_min_time(results['SPEA2']['min_time_solution'])
        ],
        'Distance for Min Time': [
            results['NSGA-II']['min_time_solution'][1] if results['NSGA-II']['min_time_solution'] is not None else float('inf'),
            results['MOEAD']['min_time_solution'][1] if results['MOEAD']['min_time_solution'] is not None else float('inf'),
            results['SPEA2']['min_time_solution'][1] if results['SPEA2']['min_time_solution'] is not None else float('inf')
        ],
        'Priority for Min Time': [
            -results['NSGA-II']['min_time_solution'][2] if results['NSGA-II']['min_time_solution'] is not None else -float('inf'),
            -results['MOEAD']['min_time_solution'][2] if results['MOEAD']['min_time_solution'] is not None else -float('inf'),
            -results['SPEA2']['min_time_solution'][2] if results['SPEA2']['min_time_solution'] is not None else -float('inf')
        ]
    })

    # pick best (same heuristic as before)
    best_time_algo = perf.loc[perf['Min Delivery Time'].idxmin()]['Algorithm']
    best_time_value = perf['Min Delivery Time'].min()
    best_priority_algo = perf.loc[perf['Priority for Min Time'].idxmax()]['Algorithm']
    best_priority_value = perf['Priority for Min Time'].max()

    if best_time_algo == best_priority_algo:
        best_overall = best_time_algo
        scores = None
    else:
        ns = results['NSGA-II']['min_time_solution'] if results['NSGA-II']['min_time_solution'] is not None else [float('inf'), float('inf'), -float('inf')]
        mo = results['MOEAD']['min_time_solution'] if results['MOEAD']['min_time_solution'] is not None else [float('inf'), float('inf'), -float('inf')]
        sp = results['SPEA2']['min_time_solution'] if results['SPEA2']['min_time_solution'] is not None else [float('inf'), float('inf'), -float('inf')]
        def score(sol):
            if sol is None:
                return float('inf')
            return sol[0] / (best_time_value if best_time_value > 0 else 1) - sol[2] / (best_priority_value if best_priority_value != 0 else 1)
        scores = {'NSGA-II': score(ns), 'MOEAD': score(mo), 'SPEA2': score(sp)}
        best_overall = min(scores, key=scores.get)

    out = {
        'results': results,
        'performance_df': perf,
        'best_algorithm': best_overall,
        'best_time_algo': best_time_algo,
        'best_time_value': best_time_value,
        'best_priority_algo': best_priority_algo,
        'best_priority_value': best_priority_value,
        'scores': scores
    }
    return out


# ---------------------------------------------------------------------
# Summarize allocation to per-site (Option A: pick RC with max units)
# ---------------------------------------------------------------------
def summarize_best_allocation(moo_out, disaster_sites, relief_centres):
    if not moo_out:
        return None
    best_algo = moo_out['best_algorithm']
    best = moo_out['results'][best_algo]
    alloc_dict = best.get('allocation') or {}

    ds_map = {}
    for idx, ds in enumerate(disaster_sites):
        key = ds.get('id', ds.get('name', f"DS{idx}"))
        ds_map[key] = {'index': idx, 'obj': ds}

    rc_map = {}
    for idx, rc in enumerate(relief_centres):
        key = rc.get('id', rc.get('name', f"RC{idx}"))
        rc_map[key] = {'index': idx, 'obj': rc}

    ds_allocs = {k: [] for k in ds_map.keys()}
    for (rc_key, ds_key), units in alloc_dict.items():
        ds_allocs.setdefault(ds_key, []).append((rc_key, units))

    per_site = []
    total_demand = 0
    total_delivered = 0
    for ds_key, info in ds_map.items():
        ds_obj = info['obj']
        demand = ds_obj.get('demand', 0)
        total_demand += demand
        alloc_list = ds_allocs.get(ds_key, [])
        delivered = sum(u for (_, u) in alloc_list)
        total_delivered += delivered
        shortage = max(demand - delivered, 0)
        if alloc_list:
            rc_with_max = max(alloc_list, key=lambda x: x[1])
            assigned_rc_key, assigned_units = rc_with_max
            rc_obj = rc_map.get(assigned_rc_key, {}).get('obj')
            if rc_obj:
                distance_km, delivery_time_min = haversine_distance_time(rc_obj['lat'], rc_obj['lon'], ds_obj['lat'], ds_obj['lon'])
                assigned_rc_name = rc_obj.get('name', assigned_rc_key)
            else:
                distance_km, delivery_time_min = 0.0, 0.0
                assigned_rc_name = assigned_rc_key
        else:
            assigned_rc_name = "None"
            distance_km, delivery_time_min = 0.0, 0.0

        fulfillment_pct = round((delivered / demand) * 100, 1) if demand > 0 else 0.0
        per_site.append({
            "disaster_site": ds_obj.get('name', ds_key),
            "area": ds_obj.get('area', 'Unknown'),
            "people_affected": ds_obj.get('people_affected', 0),
            "severity_level": ds_obj.get('severity_level', 0),
            "demand": demand,
            "delivered": delivered,
            "shortage": shortage,
            "assigned_rc": assigned_rc_name,
            "distance_km": round(distance_km, 2),
            "delivery_time_min": round(delivery_time_min, 2),
            "fulfillment_pct": fulfillment_pct
        })

    total_shortage = max(total_demand - total_delivered, 0)
    avg_fulfillment = round(sum(p['fulfillment_pct'] for p in per_site) / len(per_site), 1) if per_site else 0.0
    df = pd.DataFrame(per_site)
    return {
        "per_site": per_site,
        "total_demand": total_demand,
        "total_delivered": total_delivered,
        "total_shortage": total_shortage,
        "avg_fulfillment": avg_fulfillment,
        "df": df
    }

# ---------------------------------------------------------------------
# Map & UI helpers
# ---------------------------------------------------------------------
def create_base_map(center=DEFAULT_CENTER, zoom=12):
    return folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap', control_scale=True)

def render_map(disaster_sites=None, relief_centres=None, height=600):
    m = create_base_map()
    if disaster_sites:
        for idx, ds in enumerate(disaster_sites):
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{ds.get('name', f'Site {idx+1}')}</b><br>
                <b>Area:</b> {ds.get('area', 'Unknown')}<br>
                <b>People Affected:</b> {ds.get('people_affected', 0)}<br>
                <b>Severity:</b> {ds.get('severity_level', 0)}/10<br>
                <b>Demand:</b> {ds.get('demand', 0)} units
            </div>
            """
            folium.Marker(location=[ds["lat"], ds["lon"]],
                          popup=folium.Popup(popup_html, max_width=250),
                          icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                          tooltip=ds.get('name', f'Site {idx+1}')).add_to(m)
    if relief_centres:
        for rc in relief_centres:
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{rc['name']}</b><br>
                <b>ID:</b> {rc['id']}<br>
                <b>Supply:</b> {rc.get('supply', 0)} units<br>
                <b>Source:</b> {rc.get('source', 'Unknown')}
            </div>
            """
            color = 'blue' if rc.get('source') == 'PSO' else 'green'
            icon = 'home' if rc.get('source') == 'PSO' else 'plus'
            folium.Marker(location=[rc["lat"], rc["lon"]],
                          popup=folium.Popup(popup_html, max_width=250),
                          icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                          tooltip=rc['name']).add_to(m)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, width=None, height=height, returned_objects=["last_clicked"])
    return map_data

def add_disaster_site(lat, lon, name, area, people_affected, severity_level, time_since_last_response, demand):
    """
    IMPORTANT: add 'priority' since pso_core.fitness_function expects site['priority']
    We'll map priority = severity_level (1-10).
    """
    ds_id = len(st.session_state.disaster_sites)
    ds = {
        "id": f"DS{ds_id}",
        "name": name if name else f"Site {ds_id+1}",
        "lat": lat,
        "lon": lon,
        "area": area,
        "people_affected": people_affected,
        "severity_level": severity_level,
        "time_since_last_response": time_since_last_response,
        "demand": demand,
        # critical field for PSO fitness
        "priority": severity_level
    }
    st.session_state.disaster_sites.append(ds)
    return ds

def add_relief_centre(lat, lon, supply=1000):
    rc_id = len(st.session_state.relief_centres)
    rc = {
        "id": f"RC{rc_id}",
        "name": f"Manual RC {rc_id+1}",
        "lat": lat,
        "lon": lon,
        "supply": None,
        "area": "User Added",
        "source": "Manual"
    }
    st.session_state.relief_centres.append(rc)
    return rc

def delete_relief_centre(rc_id):
    st.session_state.relief_centres = [rc for rc in st.session_state.relief_centres if rc['id'] != rc_id]

# ---------------------------------------------------------------------
# Sidebar & forms
# ---------------------------------------------------------------------
def render_disaster_site_form():
    st.sidebar.markdown("### 📍 Disaster Site Details")
    with st.sidebar.form("disaster_site_form"):
        name = st.text_input("Site Name", value=f"Site {len(st.session_state.disaster_sites) + 1}")
        area = st.text_input("Area/Locality", value="")
        people_affected = st.number_input("People Affected", min_value=1, max_value=10000, value=100)
        severity_level = st.slider("Severity Level (1-10)", min_value=1, max_value=10, value=5)
        time_since_last_response = st.number_input("Hours Since Last Response", min_value=0, max_value=168, value=1)
        demand = st.number_input("Resource Demand (units)", min_value=1, max_value=10000, value=500)
        submitted = st.form_submit_button("✅ Add Disaster Site")
        if submitted and st.session_state.last_click:
            lat = st.session_state.last_click['lat']
            lon = st.session_state.last_click['lng']
            add_disaster_site(lat, lon, name, area, people_affected, severity_level, time_since_last_response, demand)
            st.session_state.last_click = None
            st.success(f"Added: {name}")
            st.rerun()

def render_sidebar():
    st.sidebar.title("🚨 Disaster Relief System")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Algorithm/Pipeline note:** PSO logic from `/mnt/data/pso_core.py`")
    st.sidebar.markdown("---")
    steps = {
        'setup': '1️⃣ Setup',
        'marking': '2️⃣ Mark Sites',
        'pso_run': '3️⃣ PSO Optimization',
        'editing': '4️⃣ Edit Relief Centres',
        'nsga_run': '5️⃣ Multi-Algorithm Allocation',
        'results': '✅ Results'
    }
    current = st.session_state.current_step
    st.sidebar.markdown(f"**Current Step:** {steps.get(current, 'Unknown')}")

    if current == 'setup':
        st.sidebar.markdown("### 🎯 Setup")
        num_sites = st.sidebar.number_input("How many disaster sites?", min_value=1, max_value=20, value=3, key="num_sites_input")
        num_rc = st.sidebar.number_input("How many relief centres (for PSO)?", min_value=1, max_value=20, value=5, key="num_rc_input")
        moo_mode = st.sidebar.selectbox("MOO Mode:", ["Full", "Quick"], key="moo_mode_select")
        if st.sidebar.button("🚀 Start Marking Sites"):
            st.session_state.num_disaster_sites = int(num_sites)
            st.session_state.num_relief_centres = int(num_rc)
            st.session_state.current_step = 'marking'
            st.session_state.moo_mode = moo_mode
            st.rerun()
    elif current == 'marking':
        st.sidebar.markdown("### 📍 Mark Disaster Sites")
        st.sidebar.markdown(f"**Progress:** {len(st.session_state.disaster_sites)}/{st.session_state.num_disaster_sites}")
        if st.session_state.last_click:
            render_disaster_site_form()
        else:
            st.sidebar.info("Double-click on the map to select coordinates for a disaster site.")
        if len(st.session_state.disaster_sites) >= st.session_state.num_disaster_sites:
            if st.sidebar.button("▶️ Run PSO Algorithm"):
                st.session_state.current_step = 'pso_run'
                st.rerun()
    elif current == 'pso_run':
        st.sidebar.markdown("### 🔄 PSO Running...")
        st.sidebar.info("PSO proposes relief centre locations. After completion you can edit them.")
    elif current == 'editing':
        st.markdown("### 📦 Enter Supply for Each Relief Centre")

        for rc in st.session_state.relief_centres:
            default_supply = rc.get("supply") or 0
            rc["supply"] = st.number_input(
                f"Supply for {rc['name']} ({rc['area']})",
                min_value=0,
                max_value=10000,
                value=default_supply,
                key=f"supply_{rc['id']}"
            )

        st.sidebar.markdown("### ✏️ Review Relief Centres")
        satisfied = st.sidebar.radio("Are you satisfied with the relief centres?", options=["Select...", "Yes", "No"], key="satisfaction_radio")
        if satisfied == "Yes":
            st.session_state.user_satisfied = True
            if st.sidebar.button("▶️ Run Allocation (NSGA-II + MOEA/D + SPEA2)"):
                st.session_state.current_step = 'nsga_run'
                st.rerun()
        elif satisfied == "No":
            st.session_state.user_satisfied = False
            st.sidebar.markdown("#### Edit options")
            edit_option = st.sidebar.radio("Choose action:", ["Add Relief Centre", "Delete Relief Centre"])
            if edit_option == "Add Relief Centre":
                st.sidebar.info("Double-click the map to place a manual relief centre.")
                if st.session_state.last_click:
                    supply = st.sidebar.number_input("Supply (units)", min_value=1, max_value=10000, value=1000)
                    if st.sidebar.button("➕ Add Relief Centre"):
                        add_relief_centre(st.session_state.last_click['lat'], st.session_state.last_click['lng'], supply)
                        st.session_state.last_click = None
                        st.success("Relief centre added.")
                        st.rerun()
            else:
                if st.session_state.relief_centres:
                    rc_options = [f"{rc['id']} - {rc['name']}" for rc in st.session_state.relief_centres]
                    sel = st.sidebar.selectbox("Select relief centre to delete:", rc_options)
                    if st.sidebar.button("🗑️ Delete"):
                        rc_id = sel.split(" - ")[0]
                        delete_relief_centre(rc_id)
                        st.success("Deleted.")
                        st.rerun()
                else:
                    st.sidebar.warning("No relief centres to delete.")
    elif current == 'nsga_run':
        st.sidebar.markdown("### 🔄 Allocation Running")
        st.sidebar.info("Running NSGA-II, MOEA/D (unconstrained), SPEA2. This can take minutes.")
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------
def display_results():
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.disaster_sites:
            st.markdown("### 🔴 Disaster Sites")
            df_ds = pd.DataFrame(st.session_state.disaster_sites)
            st.dataframe(df_ds[['name', 'area', 'people_affected', 'severity_level', 'demand']], use_container_width=True)
    with c2:
        if st.session_state.relief_centres:
            st.markdown("### 🔵 Relief Centres")
            df_rc = pd.DataFrame(st.session_state.relief_centres)
            st.dataframe(df_rc[['name', 'area', 'supply', 'source']], use_container_width=True)

def display_moo_summary(moo_out):
    if not moo_out:
        return
    st.markdown("## 🧾 Multi-Algorithm Comparison")
    perf = moo_out['performance_df']
    st.dataframe(perf, use_container_width=True)
    st.markdown(f"**Best Algorithm (heuristic):** {moo_out['best_algorithm']}")
    st.markdown("---")
    st.markdown("### 📌 Why chosen")
    st.write(f"- Best (min total delivery time): **{moo_out['best_time_algo']}** with **{moo_out['best_time_value']:.2f} minutes**")
    st.write(f"- Best (priority fulfillment): **{moo_out['best_priority_algo']}** with **{moo_out['best_priority_value']:.4f}**")
    if moo_out.get('scores') is not None:
        score_df = pd.DataFrame.from_dict(moo_out['scores'], orient='index', columns=['score']).reset_index().rename(columns={'index':'algorithm'})
        st.write("Combined heuristic scores (lower is better):")
        st.dataframe(score_df, use_container_width=True)
    st.markdown("---")

    best_algo = moo_out['best_algorithm']
    best = moo_out['results'][best_algo]
    st.markdown(f"### 🔎 {best_algo} - Min Time Solution Summary")
    if best['min_time_solution'] is None:
        st.warning("No solution found for best algorithm (possible infeasibility).")
    else:
        st.write("Min Time (minutes):", best['min_time_solution'][0])
        st.write("Distance for that solution (km):", best['min_time_solution'][1])
        st.write("Priority fulfillment (score):", -best['min_time_solution'][2])
        alloc = best.get('allocation') or {}
        if alloc:
            alloc_df = pd.DataFrame([{"relief_center": k[0], "disaster_site": k[1], "allocated_units": v} for k, v in alloc.items()])
            st.markdown("#### Raw Allocations (RC -> DS -> units)")
            st.dataframe(alloc_df, use_container_width=True)
        else:
            st.write("No allocations in chosen min-time solution (possible infeasibility).")

    summary = summarize_best_allocation(moo_out, st.session_state.disaster_sites, st.session_state.relief_centres)
    if not summary:
        st.warning("Could not build site summary.")
        return

    st.markdown("## 📋 Disaster Site Delivery Summary (Best solution)")
    st.dataframe(summary['df'], use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Demand", f"{summary['total_demand']:,} units")
    with c2:
        st.metric("Total Delivered", f"{summary['total_delivered']:,} units")
    with c3:
        st.metric("Total Shortage", f"{summary['total_shortage']:,} units")
    with c4:
        st.metric("Avg Fulfillment", f"{summary['avg_fulfillment']}%")

    st.markdown("---")
    st.markdown("### 📊 Demand vs Delivered")
    fig = go.Figure()
    ds_names = summary['df']['disaster_site'].tolist()
    fig.add_trace(go.Bar(name='Demand', x=ds_names, y=summary['df']['demand'].tolist()))
    fig.add_trace(go.Bar(name='Delivered', x=ds_names, y=summary['df']['delivered'].tolist()))
    fig.update_layout(title='Demand vs Delivered by Disaster Site', barmode='group', xaxis_title='Disaster Site', yaxis_title='Units')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🥧 Overall Fulfillment")
    pie_fig = px.pie(values=[summary['total_delivered'], summary['total_shortage']], names=['Fulfilled', 'Shortage'], title='Overall Fulfillment', color_discrete_sequence=['green', 'red'])
    st.plotly_chart(pie_fig, use_container_width=True)

    # CSV download
    csv = summary['df'].to_csv(index=False)
    st.download_button("⬇️ Download per-site summary (CSV)", data=csv, file_name="per_site_summary.csv", mime="text/csv")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Earthquake Relief System (MOO)", page_icon="🚨", layout="wide")
    init_session_state()
    render_sidebar()

    st.title("🚨 Earthquake Disaster Site & Relief Centre Optimization (PSO + MOO)")
    st.markdown("---")
    st.markdown("**Algorithm file path (uploaded):** `/mnt/data/pso_core.py`")

    if st.session_state.current_step == 'setup':
        st.info("Set # disaster sites and # relief centres in the sidebar to begin")
        render_map()
    elif st.session_state.current_step == 'marking':
        st.info(f"Click on the map to mark disaster sites ({len(st.session_state.disaster_sites)}/{st.session_state.num_disaster_sites})")
        map_data = render_map(disaster_sites=st.session_state.disaster_sites)
        if map_data and map_data.get('last_clicked'):
            st.session_state.last_click = map_data['last_clicked']
            st.rerun()
        display_results()
    elif st.session_state.current_step == 'pso_run':
        st.markdown("### 🔄 Running PSO Algorithm...")
        with st.spinner("Running PSO..."):
            num_rc = st.session_state.num_relief_centres if st.session_state.num_relief_centres else min(5, len(st.session_state.disaster_sites))
            relief_centres = run_pso_algorithm(st.session_state.disaster_sites, num_relief_centers=num_rc)
            st.session_state.relief_centres = relief_centres
            st.session_state.pso_completed = True
            st.session_state.current_step = 'editing'
        st.success("✅ PSO completed!")
        st.rerun()
    elif st.session_state.current_step == 'editing':
        st.success("✅ PSO completed — review or edit relief centres.")
        map_data = render_map(disaster_sites=st.session_state.disaster_sites, relief_centres=st.session_state.relief_centres)
        if map_data and map_data.get('last_clicked') and not st.session_state.user_satisfied:
            st.session_state.last_click = map_data['last_clicked']
            st.rerun()
        display_results()
    elif st.session_state.current_step == 'nsga_run':
        st.markdown("### 🔄 Running NSGA-II, MOEA/D & SPEA2")
        st.info("This will run three algorithms — may take several minutes.")
        ds_copy = [dict(ds) for ds in st.session_state.disaster_sites]
        rc_copy = [dict(rc) for rc in st.session_state.relief_centres]
        if not rc_copy:
            st.warning("No relief centres — run PSO first.")
        else:
            with st.spinner("Running allocation algorithms..."):
                moo_out = run_moo_algorithms(rc_copy, ds_copy, mode=st.session_state.moo_mode)
                if moo_out is None:
                    st.error("Multi-objective run failed (pymoo missing or other error).")
                else:
                    st.session_state.moo_results = moo_out
                    st.session_state.current_step = 'results'
                    st.success("✅ Allocation algorithms completed.")
                    st.rerun()
    elif st.session_state.current_step == 'results':
        st.markdown("### ✅ Results")
        if st.session_state.moo_results:
            display_moo_summary(st.session_state.moo_results)
            render_map(disaster_sites=st.session_state.disaster_sites, relief_centres=st.session_state.relief_centres)
        else:
            st.warning("No results — run allocation algorithms first.")

if __name__ == "__main__":
    main()
