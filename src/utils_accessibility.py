"""
================================================================================
Accessibility Analysis Utilities (utils_accessibility.py)
================================================================================

This module provides functions for calculating location-based and equity-based
accessibility measures:
    - Contour (Isochrone) Accessibility
    - Gravity (Potential) Accessibility
    - Gini Coefficient (for measuring spatial inequality)

It relies on network analysis capabilities (like shortest path calculation).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from utils_sta import mc_get_skim_matrix


# =============================================================================
# LOCATION-BASED ACCESSIBILITY MEASURES
# =============================================================================

def mc_get_skim_matrix_all_to_all(edges_gdf, zone_ids, cost_field='free_flow_time', engine='aeq'):
    """
    Computes the shortest path cost (skim) for all-to-all zones.

    Args:
        edges_gdf: GeoDataFrame with network edges (must have a_node, b_node, cost_field).
        zone_ids: List of zone/centroid IDs (must be sequential 1..N).
        cost_field: Column in edges_gdf to use as travel cost.
        engine: 'aeq' for AequilibraE (default, faster) or 'netx' for NetworkX.

    Returns:
        DataFrame with columns: origin, destination, cost (cost is travel time).
    """
    # Create dummy OD DataFrame with all zone pairs
    od_pairs = [(o, d) for o in zone_ids for d in zone_ids]
    od_gdf = pd.DataFrame(od_pairs, columns=['origin', 'destination'])
    od_gdf['demand'] = 1  # Dummy demand required by mc_get_skim_matrix

    return mc_get_skim_matrix(edges_gdf, od_gdf, cost_field=cost_field, engine=engine)


def calculate_contour_accessibility(skim_df, opportunity_df, threshold_T, verbose=False):
    """
    Calculates Location-based Contour Accessibility (Isochrone).

    Formula: $A_{i}=\Sigma_{j}O_{j}\cdot I(t_{ij}\le T)$
    A zone is 'reachable' if the travel time $t_{ij}$ is less than the threshold $T$.

    Args:
        skim_df: DataFrame with shortest path travel costs (origin, destination, cost).
        opportunity_df: DataFrame with opportunities (zone_id, opportunities).
        threshold_T: Travel time threshold (T) in minutes.

    Returns:
        Series: Accessibility value $A_i$ for each origin zone.
    """
    if verbose:
        print(f"  Calculating Contour Accessibility (Threshold T = {threshold_T} min)...")

    # 1. Merge skim matrix with opportunities (O_j)
    merged = skim_df.merge(
        opportunity_df.rename(columns={'zone_id': 'destination', 'opportunities': 'O_j'}),
        on='destination', how='left'
    )

    # Fill NaN opportunities with 0 (zones without opportunities)
    merged['O_j'] = merged['O_j'].fillna(0)

    # 2. Calculate the Indicator Function I(t_ij <= T)
    # The 'cost' column represents travel time $t_{ij}$. It is 1 if cost <= T, 0 otherwise.
    merged['is_reachable'] = (merged['cost'] <= threshold_T).astype(int)

    # 3. Calculate the accessible opportunities
    merged['accessible_opportunities'] = merged['O_j'] * merged['is_reachable']

    # 4. Sum up for each origin (A_i)
    accessibility = merged.groupby('origin')['accessible_opportunities'].sum()

    return accessibility.rename('A_contour')


def calculate_gravity_accessibility(skim_df, opportunity_df, decay_beta, verbose=False):
    """
    Calculates Location-based Gravity Accessibility (Potential).

    Formula: $A_{i}=\Sigma_{j}D_{j}\cdot e^{-\beta\cdot c_{ij}}$
    The accessibility decays exponentially with travel cost $c_{ij}$ (disutility).

    Args:
        skim_df: DataFrame with shortest path travel costs (origin, destination, cost, i.e., $c_{ij}$).
        opportunity_df: DataFrame with opportunities (zone_id, opportunities, i.e., $D_j$).
        decay_beta: Cost Sensitivity Parameter (beta, $\beta$) for exponential decay.

    Returns:
        Series: Accessibility value $A_i$ for each origin zone.
    """
    if verbose:
        print(f"  Calculating Gravity Accessibility (Decay Beta = {decay_beta})...")

    # 1. Merge skim matrix with opportunities (D_j)
    merged = skim_df.merge(
        opportunity_df.rename(columns={'zone_id': 'destination', 'opportunities': 'D_j'}),
        on='destination', how='left'
    )

    # Fill NaN opportunities with 0
    merged['D_j'] = merged['D_j'].fillna(0)

    # Convert infinite cost to a very large number (results in exp(...) near zero)
    merged['c_ij'] = merged['cost'].replace(np.inf, 1000000)

    # 2. Calculate the decay function: $\text{exp}(-\beta\cdot c_{ij})$
    merged['decay_factor'] = np.exp(-decay_beta * merged['c_ij'])

    # 3. Calculate the distance-weighted opportunities
    merged['weighted_opportunities'] = merged['D_j'] * merged['decay_factor']

    # 4. Sum up for each origin (A_i)
    accessibility = merged.groupby('origin')['weighted_opportunities'].sum()

    return accessibility.rename('A_gravity')


# =============================================================================
# EQUITY MEASURES
# =============================================================================

def calculate_gini_coefficient(accessibility_values, verbose=False):
    """
    Calculates the Gini Coefficient for a set of accessibility values.

    Measures the spatial inequality of accessibility across zones.
    0 = Perfect equality (all zones have the same accessibility).
    1 = Perfect inequality (one zone has all the accessibility).

    Formula: $G=\frac{\Sigma_{i}(2i-n-1)\cdot A_{i}}{n\Sigma A_{i}}$

    Args:
        accessibility_values: A Series or array of accessibility values ($A_i$).

    Returns:
        float: The Gini Coefficient (0 = perfect equality, 1 = perfect inequality).
    """
    if verbose:
        print("  Calculating Gini Coefficient...")

    # Ensure all values are non-negative and finite
    A_i = np.asarray(accessibility_values)
    A_i[np.isnan(A_i)] = 0
    A_i = np.maximum(A_i, 0)

    if np.sum(A_i) == 0:
        return 0.0

    # 1. Sort the accessibility values ($A_i$ in ascending order)
    A_i_sorted = np.sort(A_i)
    n = len(A_i_sorted)

    # 2. Calculate the index factor $(2i - n - 1)$
    i_indices = np.arange(1, n + 1)
    index_factor = (2 * i_indices) - n - 1

    # 3. Calculate the numerator: $\Sigma_{i}(2i-n-1)\cdot A_{i}$
    numerator = np.sum(index_factor * A_i_sorted)

    # 4. Calculate the denominator: $n\Sigma A_{i}$
    denominator = n * np.sum(A_i)

    # 5. Calculate Gini
    gini = numerator / denominator

    return gini


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================


def plot_accessibility_map(zones_gdf, accessibility_series, mode_name, measure_name, color_map='PiYG', verbose=False):
    """
    Plots the accessibility values on the zone map (single plot).

    (Network plotting logic removed to avoid incompatibility with utils_sta.plot_network)
    """
    if verbose:
        print(f"\n  Plotting {mode_name} {measure_name} map...")

    # 1. Prepare Accessibility Data
    plot_data = zones_gdf.merge(
        accessibility_series.to_frame(name='accessibility_value'),
        left_on='id',
        right_index=True,
        how='left'
    )
    max_finite = plot_data['accessibility_value'].replace([np.inf, -np.inf], np.nan).max()
    plot_data['accessibility_value'] = plot_data['accessibility_value'].replace(np.inf, max_finite).fillna(0)

    # 2. Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Use RdYlGn_r: Low Access (Red) = Bad, High Access (Green) = Good
    plot_data.plot(
        column='accessibility_value',
        ax=ax,
        legend=True,
        cmap=color_map,
        edgecolor='black',
        linewidth=0.5,
        legend_kwds={'label': f'{measure_name} Value'},
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///"
        }
    )

    ax.set_title(f'Accessibility: {mode_name.title()} - {measure_name}', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()

    return fig, ax


def plot_accessibility_map_with_basemap(zones_gdf, accessibility_series, mode_name, measure_name,
                                        basemap_source=None, alpha=0.7, color_map='PiYG', verbose=False):
    """
    Plots accessibility values on a zone map with a contextily basemap.

    Args:
        zones_gdf: GeoDataFrame with zone geometries (must have 'id' column and valid CRS).
        accessibility_series: Series with accessibility values indexed by zone id.
        mode_name: Name of transport mode (for title).
        measure_name: Name of accessibility measure (for title).
        basemap_source: Contextily basemap provider (default: CartoDB.DarkMatter).
        alpha: Transparency of zone polygons (0-1, default 0.7).

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """
    if zones_gdf.crs is None:
        zones_gdf.set_crs("EPSG:4326", inplace=True)

    if basemap_source is None:
        basemap_source = cx.providers.CartoDB.Positron

    if verbose:
        print(f"\n  Plotting {mode_name} {measure_name} map with basemap...")

    # Prepare data
    plot_data = zones_gdf.copy()
    plot_data = plot_data.merge(
        accessibility_series.to_frame(name='accessibility_value'),
        left_on='id',
        right_index=True,
        how='left'
    )

    # Handle infinite and missing values
    max_finite = plot_data['accessibility_value'].replace([np.inf, -np.inf], np.nan).max()
    plot_data['accessibility_value'] = plot_data['accessibility_value'].replace(np.inf, max_finite).fillna(0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot zones with transparency so basemap shows through
    plot_data.plot(
        column='accessibility_value',
        ax=ax,
        legend=True,
        cmap=color_map,
        edgecolor='black',
        linewidth=0.5,
        alpha=alpha,
        legend_kwds={'label': f'{measure_name} Value'},
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///"
        }
    )

    
    # Add basemap after plotting zones
    cx.add_basemap(
        ax,
        crs=zones_gdf.crs.to_string(),
        source=basemap_source
    )

    ax.set_title(f'Accessibility: {mode_name.title()} - {measure_name}', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()

    return fig, ax

