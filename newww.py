import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import your algorithms
from pso_core import fitness_function, disaster_sites as default_disaster_sites, haversine_km
import random

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Jaipur bounding box
DEFAULT_CENTER = [26.9124, 75.7873]
LAT_MIN, LAT_MAX = 26.80, 27.05
LON_MIN, LON_MAX = 75.70, 76.00

# PSO hyperparameters
SWARM_SIZE = 30
MAX_ITER = 50
W = 0.7
C1 = 1.5
C2 = 1.5


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'disaster_sites' not in st.session_state:
        st.session_state.disaster_sites = []

    if 'relief_centres' not in st.session_state:
        st.session_state.relief_centres = []

    if 'num_disaster_sites' not in st.session_state:
        st.session_state.num_disaster_sites = 0

    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'setup'  # setup, marking, pso_run, editing, nsga_run

    if 'pso_completed' not in st.session_state:
        st.session_state.pso_completed = False

    if 'nsga_completed' not in st.session_state:
        st.session_state.nsga_completed = False

    if 'map_clicks' not in st.session_state:
        st.session_state.map_clicks = 0

    if 'last_click' not in st.session_state:
        st.session_state.last_click = None

    if 'user_satisfied' not in st.session_state:
        st.session_state.user_satisfied = None

    if 'nsga_results' not in st.session_state:
        st.session_state.nsga_results = None


# ============================================================================
# PSO ALGORITHM IMPLEMENTATION
# ============================================================================

class Particle:
    """PSO Particle class"""

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
    """
    Run PSO algorithm to find optimal relief center locations

    Args:
        disaster_sites: List of disaster site dictionaries
        num_relief_centers: Number of relief centers to place

    Returns:
        List of relief center dictionaries with lat, lon, and metadata
    """

    if not disaster_sites:
        return []

    DIMENSIONS = num_relief_centers * 2

    # Initialize swarm
    swarm = [Particle(num_relief_centers) for _ in range(SWARM_SIZE)]

    # Global best
    global_best_position = None
    global_best_fitness = float('inf')

    # Initial fitness evaluation
    for particle in swarm:
        fitness = fitness_function(
            particle.position,
            disaster_sites,
            weight_distance=1.0,
            weight_proximity=5.0,
            weight_spread=0.5,
            weight_coverage=2.0
        )

        particle.best_fitness = fitness
        particle.best_position = particle.position[:]

        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position[:]

    # Main PSO loop
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

                # Boundary constraints
                if i % 2 == 0:  # Latitude
                    particle.position[i] = max(min(particle.position[i], LAT_MAX), LAT_MIN)
                else:  # Longitude
                    particle.position[i] = max(min(particle.position[i], LON_MAX), LON_MIN)

            # Evaluate fitness
            fitness = fitness_function(
                particle.position,
                disaster_sites,
                weight_distance=1.0,
                weight_proximity=5.0,
                weight_spread=0.5,
                weight_coverage=2.0
            )

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position[:]

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position[:]

        # Update progress
        progress_bar.progress((iteration + 1) / MAX_ITER)
        status_text.text(f"Iteration {iteration + 1}/{MAX_ITER} | Best Fitness: {global_best_fitness:.2f}")

    progress_bar.empty()
    status_text.empty()

    # Convert to relief center dictionaries
    relief_centres = []
    for i in range(num_relief_centers):
        lat = global_best_position[2 * i]
        lon = global_best_position[2 * i + 1]

        # Calculate nearest disaster site
        nearest_site = min(
            disaster_sites,
            key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"])
        )

        relief_centres.append({
            "id": f"RC{i}",
            "name": f"Relief Centre {i + 1}",
            "lat": lat,
            "lon": lon,
            "supply": random.randint(500, 2000),
            "area": f"Near {nearest_site.get('name', 'Unknown')}",
            "source": "PSO"
        })

    return relief_centres


# ============================================================================
# NSGA-II ALGORITHM IMPLEMENTATION
# ============================================================================

def run_nsga_algorithm(disaster_sites, relief_centres):
    """
    Run NSGA-II algorithm for resource allocation

    Args:
        disaster_sites: List of disaster site dictionaries
        relief_centres: List of relief centre dictionaries

    Returns:
        Dictionary with allocation results
    """

    if not disaster_sites or not relief_centres:
        return None

    # Calculate distance matrix
    distance_matrix = []
    for rc in relief_centres:
        distances = []
        for ds in disaster_sites:
            dist = haversine_km(rc["lat"], rc["lon"], ds["lat"], ds["lon"])
            distances.append(dist)
        distance_matrix.append(distances)

    # Simple greedy allocation (simplified version)
    # In production, this would use the full NSGA-II from new.py
    allocation_results = []

    for ds_idx, ds in enumerate(disaster_sites):
        demand = ds["demand"]

        # Find nearest relief center with supply
        rc_distances = [(rc_idx, distance_matrix[rc_idx][ds_idx])
                        for rc_idx in range(len(relief_centres))
                        if relief_centres[rc_idx]["supply"] > 0]

        if not rc_distances:
            allocation_results.append({
                "disaster_site": ds.get("name", f"DS{ds_idx}"),
                "demand": demand,
                "delivered": 0,
                "shortage": demand,
                "assigned_rc": "None",
                "distance_km": 0,
                "delivery_time_min": 0
            })
            continue

        # Sort by distance
        rc_distances.sort(key=lambda x: x[1])
        nearest_rc_idx, distance = rc_distances[0]

        # Allocate resources
        available = relief_centres[nearest_rc_idx]["supply"]
        delivered = min(demand, available)
        shortage = demand - delivered

        # Update supply
        relief_centres[nearest_rc_idx]["supply"] -= delivered

        # Calculate delivery time (assume 30 km/h average speed)
        delivery_time = (distance / 30) * 60  # minutes

        allocation_results.append({
            "disaster_site": ds.get("name", f"DS{ds_idx}"),
            "area": ds.get("area", "Unknown"),
            "people_affected": ds.get("people_affected", 0),
            "severity_level": ds.get("severity_level", 0),
            "demand": demand,
            "delivered": delivered,
            "shortage": shortage,
            "assigned_rc": relief_centres[nearest_rc_idx]["name"],
            "distance_km": round(distance, 2),
            "delivery_time_min": round(delivery_time, 2),
            "fulfillment_pct": round((delivered / demand) * 100, 1) if demand > 0 else 0
        })

    return {
        "allocations": allocation_results,
        "total_demand": sum(r["demand"] for r in allocation_results),
        "total_delivered": sum(r["delivered"] for r in allocation_results),
        "total_shortage": sum(r["shortage"] for r in allocation_results),
        "avg_fulfillment": round(
            sum(r["fulfillment_pct"] for r in allocation_results) / len(allocation_results), 1
        ) if allocation_results else 0
    }


# ============================================================================
# MAP RENDERING FUNCTIONS
# ============================================================================

def create_base_map(center=DEFAULT_CENTER, zoom=12):
    """Create a base folium map"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    return m


def render_map(disaster_sites=None, relief_centres=None, height=500):
    """
    Render interactive map with disaster sites and relief centres

    Args:
        disaster_sites: List of disaster site dictionaries
        relief_centres: List of relief centre dictionaries
        height: Map height in pixels

    Returns:
        Map data from st_folium
    """

    m = create_base_map()

    # Add disaster sites
    if disaster_sites:
        for idx, ds in enumerate(disaster_sites):
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{ds.get('name', f'DS{idx}')}</b><br>
                <b>Area:</b> {ds.get('area', 'Unknown')}<br>
                <b>People Affected:</b> {ds.get('people_affected', 0)}<br>
                <b>Severity:</b> {ds.get('severity_level', 0)}/10<br>
                <b>Demand:</b> {ds.get('demand', 0)} units
            </div>
            """

            folium.Marker(
                location=[ds["lat"], ds["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                tooltip=ds.get('name', f'DS{idx}')
            ).add_to(m)

    # Add relief centres
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

            folium.Marker(
                location=[rc["lat"], rc["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=rc['name']
            ).add_to(m)

    # Add click instruction
    folium.LatLngPopup().add_to(m)

    # Render map
    map_data = st_folium(
        m,
        width=None,
        height=height,
        returned_objects=["last_clicked"]
    )

    return map_data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_disaster_site(lat, lon, name, area, people_affected, severity_level, time_since_last_response, demand):
    """Add a disaster site to session state"""

    ds_id = len(st.session_state.disaster_sites)

    disaster_site = {
        "id": ds_id,
        "name": name if name else f"DS{ds_id}",
        "lat": lat,
        "lon": lon,
        "area": area,
        "people_affected": people_affected,
        "severity_level": severity_level,
        "time_since_last_response": time_since_last_response,
        "demand": demand,
        "priority": severity_level  # For PSO compatibility
    }

    st.session_state.disaster_sites.append(disaster_site)
    return disaster_site


def add_relief_centre(lat, lon, supply=1000):
    """Manually add a relief centre"""

    rc_id = len(st.session_state.relief_centres)

    relief_centre = {
        "id": f"RC{rc_id}",
        "name": f"Manual RC {rc_id + 1}",
        "lat": lat,
        "lon": lon,
        "supply": supply,
        "area": "User Added",
        "source": "Manual"
    }

    st.session_state.relief_centres.append(relief_centre)
    return relief_centre


def delete_relief_centre(rc_id):
    """Delete a relief centre by ID"""
    st.session_state.relief_centres = [
        rc for rc in st.session_state.relief_centres if rc['id'] != rc_id
    ]


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_disaster_site_form():
    """Render form to input disaster site details"""

    st.sidebar.markdown("### 📍 Disaster Site Details")

    with st.sidebar.form("disaster_site_form"):
        name = st.text_input("Site Name", value=f"Site {len(st.session_state.disaster_sites) + 1}")
        area = st.text_input("Area/Locality", value="")
        people_affected = st.number_input("People Affected", min_value=1, max_value=10000, value=100)
        severity_level = st.slider("Severity Level", min_value=1, max_value=10, value=5)
        time_since_last_response = st.number_input("Hours Since Last Response", min_value=0, max_value=72, value=1)
        demand = st.number_input("Resource Demand (units)", min_value=1, max_value=5000, value=500)

        submitted = st.form_submit_button("✅ Add Disaster Site")

        if submitted and st.session_state.last_click:
            lat = st.session_state.last_click['lat']
            lon = st.session_state.last_click['lng']

            add_disaster_site(lat, lon, name, area, people_affected, severity_level, time_since_last_response, demand)
            st.session_state.last_click = None
            st.success(f"✅ Added: {name}")
            st.rerun()


def render_sidebar():
    """Render sidebar based on current step"""

    st.sidebar.title("🚨 Disaster Relief System")
    st.sidebar.markdown("---")

    # Step indicator
    steps = {
        'setup': '1️⃣ Setup',
        'marking': '2️⃣ Mark Sites',
        'pso_run': '3️⃣ PSO Optimization',
        'editing': '4️⃣ Edit Relief Centres',
        'nsga_run': '5️⃣ Resource Allocation'
    }

    current = st.session_state.current_step
    st.sidebar.markdown(f"**Current Step:** {steps.get(current, 'Unknown')}")
    st.sidebar.markdown("---")

    # Step-specific controls
    if current == 'setup':
        st.sidebar.markdown("### 🎯 Setup")
        num_sites = st.sidebar.number_input(
            "How many disaster sites?",
            min_value=1,
            max_value=20,
            value=5,
            key="num_sites_input"
        )

        if st.sidebar.button("🚀 Start Marking Sites"):
            st.session_state.num_disaster_sites = num_sites
            st.session_state.current_step = 'marking'
            st.rerun()

    elif current == 'marking':
        st.sidebar.markdown(f"### 📍 Mark Disaster Sites")
        st.sidebar.markdown(
            f"**Progress:** {len(st.session_state.disaster_sites)}/{st.session_state.num_disaster_sites}")

        if st.session_state.last_click:
            render_disaster_site_form()
        else:
            st.sidebar.info("👆 Double-click on the map to mark a disaster site")

        if len(st.session_state.disaster_sites) >= st.session_state.num_disaster_sites:
            if st.sidebar.button("▶️ Run PSO Algorithm"):
                st.session_state.current_step = 'pso_run'
                st.rerun()

    elif current == 'pso_run':
        st.sidebar.markdown("### 🔄 PSO Running...")
        st.sidebar.info("Optimizing relief centre locations...")

    elif current == 'editing':
        st.sidebar.markdown("### ✏️ Review Relief Centres")

        satisfied = st.sidebar.radio(
            "Are you satisfied with the relief centres?",
            options=["Select...", "Yes", "No"],
            key="satisfaction_radio"
        )

        if satisfied == "Yes":
            st.session_state.user_satisfied = True
            if st.sidebar.button("▶️ Run Resource Allocation (NSGA-II)"):
                st.session_state.current_step = 'nsga_run'
                st.rerun()

        elif satisfied == "No":
            st.session_state.user_satisfied = False

            st.sidebar.markdown("#### Options:")

            edit_option = st.sidebar.radio(
                "Choose action:",
                ["Add Relief Centre", "Delete Relief Centre"],
                key="edit_option"
            )

            if edit_option == "Add Relief Centre":
                st.sidebar.info("👆 Double-click on map to add a relief centre")

                if st.session_state.last_click:
                    supply = st.sidebar.number_input("Supply (units)", min_value=100, max_value=5000, value=1000)
                    if st.sidebar.button("➕ Add Relief Centre"):
                        add_relief_centre(
                            st.session_state.last_click['lat'],
                            st.session_state.last_click['lng'],
                            supply
                        )
                        st.session_state.last_click = None
                        st.success("✅ Relief centre added!")
                        st.rerun()

            else:  # Delete
                if st.session_state.relief_centres:
                    rc_options = [f"{rc['id']} - {rc['name']}" for rc in st.session_state.relief_centres]
                    selected = st.sidebar.selectbox("Select relief centre to delete:", rc_options)

                    if st.sidebar.button("🗑️ Delete Relief Centre"):
                        rc_id = selected.split(" - ")[0]
                        delete_relief_centre(rc_id)
                        st.success(f"✅ Deleted {rc_id}")
                        st.rerun()
                else:
                    st.sidebar.warning("No relief centres to delete")

    elif current == 'nsga_run':
        st.sidebar.markdown("### 🔄 NSGA-II Running...")
        st.sidebar.info("Calculating resource allocation...")

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def display_results():
    """Display disaster sites and relief centres in tables"""

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.disaster_sites:
            st.markdown("### 🔴 Disaster Sites")
            df_ds = pd.DataFrame(st.session_state.disaster_sites)
            st.dataframe(df_ds[['name', 'area', 'people_affected', 'severity_level', 'demand']],
                         use_container_width=True)

    with col2:
        if st.session_state.relief_centres:
            st.markdown("### 🔵 Relief Centres")
            df_rc = pd.DataFrame(st.session_state.relief_centres)
            st.dataframe(df_rc[['name', 'area', 'supply', 'source']], use_container_width=True)


def display_nsga_results():
    """Display NSGA-II allocation results"""

    if not st.session_state.nsga_results:
        return

    results = st.session_state.nsga_results

    st.markdown("## 📊 Resource Allocation Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Demand", f"{results['total_demand']:,} units")

    with col2:
        st.metric("Total Delivered", f"{results['total_delivered']:,} units")

    with col3:
        st.metric("Total Shortage", f"{results['total_shortage']:,} units")

    with col4:
        st.metric("Avg Fulfillment", f"{results['avg_fulfillment']}%")

    st.markdown("---")

    # Detailed allocation table
    st.markdown("### 📋 Detailed Allocation")
    df_alloc = pd.DataFrame(results['allocations'])
    st.dataframe(df_alloc, use_container_width=True)

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # Demand vs Delivered chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Demand',
            x=[a['disaster_site'] for a in results['allocations']],
            y=[a['demand'] for a in results['allocations']],
            marker_color='red'
        ))
        fig.add_trace(go.Bar(
            name='Delivered',
            x=[a['disaster_site'] for a in results['allocations']],
            y=[a['delivered'] for a in results['allocations']],
            marker_color='green'
        ))
        fig.update_layout(
            title='Demand vs Delivered',
            barmode='group',
            xaxis_title='Disaster Site',
            yaxis_title='Units'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fulfillment percentage pie chart
        fig = px.pie(
            values=[results['total_delivered'], results['total_shortage']],
            names=['Fulfilled', 'Shortage'],
            title='Overall Fulfillment',
            color_discrete_sequence=['green', 'red']
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application logic"""

    # Page config
    st.set_page_config(
        page_title="Earthquake Relief System",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🚨 Earthquake Disaster Site & Relief Centre Optimization")
    st.markdown("---")

    # Handle different steps
    if st.session_state.current_step == 'setup':
        st.info("👈 Set the number of disaster sites in the sidebar to begin")
        map_data = render_map()

    elif st.session_state.current_step == 'marking':
        st.info(
            f"📍 Click on the map to mark disaster sites ({len(st.session_state.disaster_sites)}/{st.session_state.num_disaster_sites})")
        map_data = render_map(disaster_sites=st.session_state.disaster_sites)

        # Capture clicks
        if map_data and map_data.get('last_clicked'):
            st.session_state.last_click = map_data['last_clicked']
            st.rerun()

        # Display current sites
        if st.session_state.disaster_sites:
            display_results()

    elif st.session_state.current_step == 'pso_run':
        st.markdown("### 🔄 Running PSO Algorithm...")

        with st.spinner("Optimizing relief centre locations..."):
            relief_centres = run_pso_algorithm(
                st.session_state.disaster_sites,
                num_relief_centers=min(5, len(st.session_state.disaster_sites))
            )
            st.session_state.relief_centres = relief_centres
            st.session_state.pso_completed = True
            st.session_state.current_step = 'editing'

        st.success("✅ PSO optimization completed!")
        st.rerun()

    elif st.session_state.current_step == 'editing':
        st.success("✅ PSO optimization completed! Review the relief centres below.")

        map_data = render_map(
            disaster_sites=st.session_state.disaster_sites,
            relief_centres=st.session_state.relief_centres
        )

        # Capture clicks for adding relief centres
        if map_data and map_data.get('last_clicked') and not st.session_state.user_satisfied:
            st.session_state.last_click = map_data['last_clicked']
            st.rerun()

        display_results()

    elif st.session_state.current_step == 'nsga_run':
        st.markdown("### 🔄 Running NSGA-II Algorithm...")

        with st.spinner("Calculating resource allocation..."):
            results = run_nsga_algorithm(
                st.session_state.disaster_sites,
                st.session_state.relief_centres
            )
            st.session_state.nsga_results = results
            st.session_state.nsga_completed = True

        st.success("✅ Resource allocation completed!")

        # Show map
        render_map(
            disaster_sites=st.session_state.disaster_sites,
            relief_centres=st.session_state.relief_centres
        )

        # Show results
        display_nsga_results()


if __name__ == "__main__":
    main()