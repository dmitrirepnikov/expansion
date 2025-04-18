import pandas as pd
import numpy as np
import folium
import h3
import streamlit as st
from streamlit_folium import st_folium
from collections import defaultdict
from google.cloud import bigquery
import google.oauth2.credentials
from datetime import datetime, timedelta
import pytz
import os
import traceback

# Set page config
st.set_page_config(page_title="H3 Delivery Map", layout="wide")

# Initialize BigQuery client with improved error handling
@st.cache_resource
def get_bq_client():
    try:
        st.sidebar.info("Initializing BigQuery client...")
        # Check if running locally or in Streamlit Cloud
        if "gcp_service_account" in st.secrets:
            st.sidebar.info("Using Streamlit Cloud authentication")
            # Use stored secrets in Streamlit Cloud
            credentials_dict = st.secrets["gcp_service_account"]
            credentials = google.oauth2.credentials.Credentials(
                None,
                refresh_token=credentials_dict['refresh_token'],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=credentials_dict['client_id'],
                client_secret=credentials_dict['client_secret']
            )
        else:
            st.sidebar.info("Using local authentication")
            # When running locally, use default credentials
            credentials = None
        
        client = bigquery.Client(project='postmates-x', credentials=credentials)
        # Test the connection
        client.list_datasets(max_results=1)
        st.sidebar.success("✅ BigQuery connection successful!")
        return client
    except Exception as e:
        error_msg = f"❌ Error initializing BigQuery client: {str(e)}"
        st.sidebar.error(error_msg)
        st.error(error_msg)
        raise

# Function to fetch data from BigQuery with improved error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_expansion_data(limit_rows=False):
    debug_container = st.empty()
    try:
        debug_container.info("Step 1: Initializing BigQuery client...")
        client = get_bq_client()
        
        debug_container.info("Step 2: Preparing query...")
        # Prepare query - optionally with limit for testing
        if limit_rows:
            query = """
            SELECT
              city,
              coordinates AS coordinate,
              CAST(latitude AS FLOAT64) AS latitude,
              CAST(longitude AS FLOAT64) AS longitude,
              CAST(delivs_a AS FLOAT64) AS delivs_a,
              CAST(delivs_b AS FLOAT64) AS delivs_b,
              CAST(delivs_c AS FLOAT64) AS delivs_c
            FROM
              `postmates-x.dev_dmitrii.expansion_rounded`
            LIMIT 500

            UNION ALL

            SELECT
              city,
              lat_lng AS coordinate,
              CAST(latitude AS FLOAT64) AS latitude,
              CAST(longitude AS FLOAT64) AS longitude,
              CAST(demand_a AS FLOAT64) AS delivs_a,
              CAST(demand_b AS FLOAT64) AS delivs_b,
              CAST(demand_c AS FLOAT64) AS delivs_c
            FROM
              `postmates-x.dev_dmitrii.int_expansion`
            LIMIT 500
            """
        else:
            query = """
            SELECT
              city,
              coordinates AS coordinate,
              CAST(latitude AS FLOAT64) AS latitude,
              CAST(longitude AS FLOAT64) AS longitude,
              CAST(delivs_a AS FLOAT64) AS delivs_a,
              CAST(delivs_b AS FLOAT64) AS delivs_b,
              CAST(delivs_c AS FLOAT64) AS delivs_c
            FROM
              `postmates-x.dev_dmitrii.expansion_rounded`

            UNION ALL

            SELECT
              city,
              lat_lng AS coordinate,
              CAST(latitude AS FLOAT64) AS latitude,
              CAST(longitude AS FLOAT64) AS longitude,
              CAST(demand_a AS FLOAT64) AS delivs_a,
              CAST(demand_b AS FLOAT64) AS delivs_b,
              CAST(demand_c AS FLOAT64) AS delivs_c
            FROM
              `postmates-x.dev_dmitrii.int_expansion`
            """
        
        debug_container.info("Step 3: Executing query...")
        with st.spinner('Fetching data from BigQuery...'):
            try:
                query_job = client.query(query)
                debug_container.info(f"Step 4: Query job created with ID: {query_job.job_id}")
                
                debug_container.info("Step 5: Waiting for query to complete...")
                results = query_job.result()
                
                debug_container.info("Step 6: Converting results to dataframe...")
                df = results.to_dataframe()
                
                # Add refresh timestamp
                pst = pytz.timezone('America/Los_Angeles')
                refresh_time = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
                
                debug_container.success(f"✅ Data retrieved: {len(df)} rows")
                return df, refresh_time
            except Exception as e:
                error_msg = f"❌ Error in query execution: {str(e)}"
                debug_container.error(error_msg)
                st.code(traceback.format_exc())
                return pd.DataFrame(), None
    except Exception as e:
        error_msg = f"❌ Error in fetch_expansion_data: {str(e)}"
        debug_container.error(error_msg)
        st.code(traceback.format_exc())
        return pd.DataFrame(), None
    finally:
        # Clear debug container after 5 seconds
        import time
        time.sleep(5)
        debug_container.empty()

# Simplified test query function for troubleshooting
def test_bigquery_connection():
    try:
        client = get_bq_client()
        test_query = "SELECT 1 as test"
        result = client.query(test_query).result().to_dataframe()
        st.sidebar.success("✅ Test query executed successfully")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Test query failed: {str(e)}")
        st.code(traceback.format_exc())
        return False

def create_hierarchical_h3_map(df, display_resolution=5, base_resolution=11, distinct_region_colors=True, 
                               custom_eps=0.15, use_demand_colors=True, use_region_specific_scale=True,
                               census_tract_data=None, transparency_level=0.7, max_cells=10000, show_tooltips=True):
    """
    Create an H3 map that properly aggregates data across resolution levels

    Parameters:
    df - DataFrame with lat/long and delivery data
    display_resolution - The resolution to display (lower = larger hexagons)
    base_resolution - The highest resolution to use for initial point mapping
    custom_eps - DBSCAN epsilon parameter for region detection (higher = larger regions)
    use_demand_colors - If True, colors cells based on demand level (green to red); if False, uses distinct colors for regions
    census_tract_data - Optional GeoDataFrame containing census tract boundaries with population density data
    transparency_level - Base transparency level for H3 cells (higher = more opaque)

    Returns:
    folium.Map, dict of H3 cells
    """
    st.info(f"Processing {len(df)} points using hierarchical H3 aggregation...")
    st.info(f"Base resolution: {base_resolution}, Display resolution: {display_resolution}")
    if use_demand_colors:
        color_scheme_text = "region-specific demand-based" if use_region_specific_scale else "global demand-based"
    else:
        color_scheme_text = "region-based"
    st.info(f"Using {color_scheme_text} coloring")

    try:
        # Copy dataframe to avoid modifying original
        work_df = df.copy()

        # Convert coordinates to numeric - with explicit error handling
        work_df['latitude'] = pd.to_numeric(work_df['latitude'], errors='raise')
        work_df['longitude'] = pd.to_numeric(work_df['longitude'], errors='raise')

        # Convert deliveries columns to numeric
        work_df['delivs_a'] = pd.to_numeric(work_df['delivs_a'], errors='coerce').fillna(0)
        work_df['delivs_b'] = pd.to_numeric(work_df['delivs_b'], errors='coerce').fillna(0)
        work_df['delivs_c'] = pd.to_numeric(work_df['delivs_c'], errors='coerce').fillna(0)

        # 1. First, assign each point to its highest resolution H3 cell
        high_res_cells = {}
        error_count = 0
        processed_count = 0

        for idx, row in work_df.iterrows():
            try:
                # Skip rows with invalid coordinates
                if not (-90 <= row['latitude'] <= 90) or not (-180 <= row['longitude'] <= 180):
                    continue
                
                # Convert lat/long to high-resolution H3 cell
                h3_cell = h3.latlng_to_cell(row['latitude'], row['longitude'], base_resolution)
                processed_count += 1

                # Store the data for this cell
                if h3_cell not in high_res_cells:
                    high_res_cells[h3_cell] = {
                        'city': row['city'],
                        'region': row.get('region', row['city']),  # Use region if available, otherwise use city
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'delivs_a': 0,
                        'delivs_b': 0,
                        'delivs_c': 0,
                        'count': 0
                    }

                # Add demand data to the cell
                high_res_cells[h3_cell]['delivs_a'] += row['delivs_a']
                high_res_cells[h3_cell]['delivs_b'] += row['delivs_b']
                high_res_cells[h3_cell]['delivs_c'] += row['delivs_c']
                high_res_cells[h3_cell]['count'] += 1

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only show first few errors
                    st.warning(f"Error processing row {idx}: {e}")
                elif error_count == 6:
                    st.warning(f"Additional errors found. Showing only the first 5.")

        # Status update
        st.info(f"Processed {processed_count} points, created {len(high_res_cells)} high-resolution cells")
        if error_count > 0:
            st.warning(f"{error_count} points could not be processed")

        if len(high_res_cells) == 0:
            st.error("No valid H3 cells were created. Please check your data.")
            return None, {}

        # 2. Now aggregate these cells up to the display resolution
        display_cells = {}

        for h3_index, cell_data in high_res_cells.items():
            try:
                # Get the parent cell at the display resolution
                parent_cell = h3.cell_to_parent(h3_index, display_resolution)

                # Create parent cell if it doesn't exist
                if parent_cell not in display_cells:
                    display_cells[parent_cell] = {
                        'city': cell_data['city'],  # Use city from first child cell
                        'region': cell_data['region'],  # Use region from first child cell
                        'latitude': 0,
                        'longitude': 0,
                        'delivs_a': 0,
                        'delivs_b': 0,
                        'delivs_c': 0,
                        'child_cells': 0,
                        'count': 0,
                        'regions': {}  # Track all regions in this cell for majority voting
                    }

                # Aggregate data to parent cell
                display_cells[parent_cell]['delivs_a'] += cell_data['delivs_a']
                display_cells[parent_cell]['delivs_b'] += cell_data['delivs_b']
                display_cells[parent_cell]['delivs_c'] += cell_data['delivs_c']
                display_cells[parent_cell]['child_cells'] += 1
                display_cells[parent_cell]['count'] += cell_data['count']

                # Count occurrences of each region for majority voting
                region = cell_data['region']
                if region not in display_cells[parent_cell]['regions']:
                    display_cells[parent_cell]['regions'][region] = 0
                display_cells[parent_cell]['regions'][region] += cell_data['count']

                # Accumulate lat/long for averaging later
                lat, lng = h3.cell_to_latlng(h3_index)
                display_cells[parent_cell]['latitude'] += lat * cell_data['count']
                display_cells[parent_cell]['longitude'] += lng * cell_data['count']
            except Exception as e:
                st.warning(f"Error aggregating cell {h3_index}: {e}")

        st.info(f"Created {len(display_cells)} display cells at resolution {display_resolution}")

        # Calculate average lat/long and determine majority region for each display cell
        for h3_index, cell_data in display_cells.items():
            if cell_data['count'] > 0:
                cell_data['latitude'] /= cell_data['count']
                cell_data['longitude'] /= cell_data['count']
            else:
                # Fallback to cell center if no points
                lat, lng = h3.cell_to_latlng(h3_index)
                cell_data['latitude'] = lat
                cell_data['longitude'] = lng

            # Determine majority region
            if cell_data['regions']:
                cell_data['region'] = max(cell_data['regions'].items(), key=lambda x: x[1])[0]

            # Add total deliveries
            cell_data['total_delivs'] = cell_data['delivs_a'] + cell_data['delivs_b'] + cell_data['delivs_c']

        # Find the center point for the map
        try:
            all_lats = [cell['latitude'] for cell in display_cells.values()]
            all_lngs = [cell['longitude'] for cell in display_cells.values()]
            center_lat = sum(all_lats) / len(all_lats)
            center_lng = sum(all_lngs) / len(all_lngs)
        except Exception as e:
            st.warning(f"Error calculating map center: {e}")
            # Default to a reasonable center point if calculation fails
            center_lat, center_lng = 37.7749, -122.4194  # Default to San Francisco

        # Define distinct color palette for regions
        color_palette = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',  # Brewer Set1
            '#a65628', '#f781bf', '#999999', '#e6ab02', '#66a61e',  # More distinct colors
            '#8dd3c7', '#bebada', '#fb8072', '#80b1d3', '#fdb462',  # Brewer Set2
            '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5',
            '#ffed6f', '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00'
        ]

        # 3. Create the map with display cells
        m = folium.Map(location=[center_lat, center_lng], zoom_start=7, tiles='CartoDB positron')

        # Add alternative base maps
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)

        # Add census tract population density layer if provided
        if census_tract_data is not None:
            census_style_function = lambda x: {
                'fillColor': '#000000' if x['properties']['population_density'] > 5000 else
                            '#333333' if x['properties']['population_density'] > 2500 else
                            '#666666' if x['properties']['population_density'] > 1000 else
                            '#999999' if x['properties']['population_density'] > 500 else
                            '#cccccc',
                'color': '#222222',
                'weight': 0.5,
                'fillOpacity': 0.3
            }

            # Create a feature group for census tracts that can be toggled
            census_group = folium.FeatureGroup(name='Population Density (Census Tracts)')

            # Add GeoJson layer for census tracts
            folium.GeoJson(
                census_tract_data,
                name='Census Tracts',
                style_function=census_style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['GEOID', 'population_density'],
                    aliases=['Census Tract:', 'Population Density:'],
                    localize=True
                )
            ).add_to(census_group)

            # Add the census tract layer to the map
            census_group.add_to(m)

        # Get unique regions
        all_regions = set(cell['region'] for cell in display_cells.values())

        # Assign colors to regions
        region_colors = {}

        # Sort regions by name to ensure consistent coloring
        sorted_regions = sorted(all_regions)
        for i, region in enumerate(sorted_regions):
            region_colors[region] = color_palette[i % len(color_palette)]

        # Group cells by region
        region_cells = defaultdict(list)
        for h3_index, cell_data in display_cells.items():
            region_cells[cell_data['region']].append(h3_index)

        # Calculate regional min/max for demand scaling
        region_stats = {}
        for region, cell_indices in region_cells.items():
            region_values = [display_cells[idx]['delivs_a'] for idx in cell_indices]

            if not region_values:
                continue

            # Get min/max for this region
            region_min = min(region_values)
            region_max = max(region_values)

            # Make sure min and max are different to avoid division by zero
            if region_min == region_max:
                region_max = region_min + 0.01  # Small difference to avoid division by zero

            # Store min/max for this region
            region_stats[region] = {
                'min': region_min,
                'max': region_max,
                'avg': sum(region_values) / len(region_values)
            }

        # Get global min/max delivery values (for opacity scaling)
        all_values = [cell['delivs_a'] for cell in display_cells.values()]
        min_val = min(all_values) if all_values else 0
        max_val = max(all_values) if all_values else 1

        # Make sure min and max are different to avoid division by zero
        if min_val == max_val:
            max_val = min_val + 1

        # Function to get color by demand level (green to red gradient)
        def get_demand_color(value, min_val, max_val):
            # Normalize value between 0 and 1
            if max_val == min_val:
                normalized = 0.5  # Default to middle of range if min=max
            else:
                normalized = (value - min_val) / (max_val - min_val)

            # Green to red color scale
            if normalized < 0.1:
                return '#e5f5e0'  # Very light green
            elif normalized < 0.2:
                return '#c7e9c0'  # Light green
            elif normalized < 0.3:
                return '#a1d99b'  # Green
            elif normalized < 0.4:
                return '#74c476'  # Medium green
            elif normalized < 0.5:
                return '#fdbe85'  # Light orange
            elif normalized < 0.6:
                return '#fd8d3c'  # Orange
            elif normalized < 0.7:
                return '#f03b20'  # Orange-red
            elif normalized < 0.85:
                return '#bd0026'  # Red
            else:
                return '#7f0000'  # Dark red

        # Display cells on the map
        cell_count = 0
        error_count = 0

        for h3_index, cell_data in display_cells.items():
            try:
                # Get the boundary of the H3 cell
                boundary = h3.cell_to_boundary(h3_index)

                # Convert boundary to folium format
                folium_boundary = [[float(lat), float(lng)] for lat, lng in boundary]

                # Get cell region
                region = cell_data['region']

                # Choose coloring method
                if use_demand_colors:
                    if use_region_specific_scale:
                        # For region-specific demand-based coloring
                        if region in region_stats:
                            # Get region-specific stats for normalization
                            region_min = region_stats[region]['min']
                            region_max = region_stats[region]['max']
                            
                            # Ensure minimum difference to avoid division by zero or minimal color variation
                            if region_max - region_min < 0.1:
                                region_max = region_min + 0.1
                            
                            # For demand-based coloring within region, we'll use a gradient from green to red
                            demand_color = get_demand_color(cell_data['delivs_a'], region_min, region_max)
                            
                            # Use the demand color for demand-based visualization
                            color = demand_color
                        else:
                            # Fallback color
                            color = '#cccccc'
                    else:
                        # Global demand-based coloring (uses global min/max)
                        color = get_demand_color(cell_data['delivs_a'], min_val, max_val)
                    
                    # Use the passed transparency_level parameter for controlling opacity
                    opacity = transparency_level
                else:
                    # For region-based coloring with distinct colors
                    color = region_colors[region]

                    # Adjust opacity based on demand - higher demand = more opaque
                    # But scale it by the transparency_level parameter
                    normalized_demand = (cell_data['delivs_a'] - min_val) / (max_val - min_val)
                    # Reduce the max opacity to improve map visibility
                    opacity = max(0.2, min(transparency_level, 0.2 + normalized_demand * transparency_level))

                # Create popup content
                popup_html = f"""
                <div style="width: 200px; font-family: Arial, sans-serif; padding: 5px;">
                    <h4 style="margin-top: 0;">{cell_data['city']}</h4>
                    <p><b>Region:</b> {region}</p>
                    <p><b>H3 Index:</b> {h3_index}</p>
                    <p><b>Resolution:</b> {display_resolution}</p>
                    <p><b>Deliveries Type A:</b> {cell_data['delivs_a']:.2f}</p>
                    <p><b>Deliveries Type B:</b> {cell_data['delivs_b']:.2f}</p>
                    <p><b>Deliveries Type C:</b> {cell_data['delivs_c']:.2f}</p>
                    <p><b>Total Deliveries:</b> {cell_data['total_delivs']:.2f}</p>
                    <p><b>Child Cells:</b> {cell_data['child_cells']}</p>
                    <p><b>Data Points:</b> {cell_data['count']}</p>
                """

                # Add regional demand context to popup if using demand colors
                if use_demand_colors and region in region_stats:
                    region_min = region_stats[region]['min']
                    region_max = region_stats[region]['max']
                    region_avg = region_stats[region]['avg']

                    # Add regional demand context
                    popup_html += f"""
                    <hr style="margin: 5px 0;">
                    <p><b>Regional Context:</b></p>
                    <p>Min Demand: {region_min:.2f}</p>
                    <p>Avg Demand: {region_avg:.2f}</p>
                    <p>Max Demand: {region_max:.2f}</p>
                    """

                # Close the div
                popup_html += "</div>"

                # Create tooltip with the requested information
                tooltip = f"Region: {region} | Delivs A: {cell_data['delivs_a']:.2f}"

                # Add polygon to map with light black outline
                folium.Polygon(
                    locations=folium_boundary,
                    color='#333333',  # Light black color for outline
                    weight=0.5,       # Thinner outline for cleaner look on dense maps
                    fill=True,
                    fill_color=color,
                    fill_opacity=opacity,
                    tooltip=tooltip if show_tooltips else None,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(m)
                
                # Safety check - limit the number of cells to render to avoid browser performance issues
                if cell_count >= max_cells:
                    st.warning(f"Reached maximum cell limit ({max_cells}). Some cells may not be displayed. Increase the limit in Advanced Options if needed.")
                    break

                cell_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only show first few errors
                    st.warning(f"Error adding cell {h3_index} to map: {e}")
                elif error_count == 6:
                    st.warning(f"Additional errors occurred. Showing only the first 5.")

        st.info(f"Added {cell_count} cells to the map")
        if error_count > 0:
            st.warning(f"{error_count} cells could not be added to the map")

        # Add layer control
        folium.LayerControl().add_to(m)

        return m, display_cells
    
    except Exception as e:
        st.error(f"Error in map creation: {str(e)}")
        st.code(traceback.format_exc())
        return None, {}

# Main Streamlit App
def main():
    st.title("Hierarchical H3 Map with Adjustable Cell Size")
    st.markdown("Visualize delivery data with customizable H3 cell sizes")
    
    # Sidebar for controls
    st.sidebar.header("Map Settings")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        options=["BigQuery Database", "CSV Upload"],
        index=0
    )
    
    # Initialize dataframe
    df = pd.DataFrame()
    refresh_time = None
    
    # Load data based on selected source
    if data_source == "BigQuery Database":
        # Add auto-fetch option and fetch settings
        with st.sidebar.expander("Fetch Settings", expanded=True):
            auto_fetch = st.checkbox("Auto-fetch on selection", value=True)
            include_all_regions = st.checkbox("Include all global regions", value=True, 
                                              help="When enabled, fetches all regions including Asian markets")
            test_mode = st.checkbox("Test mode (limit rows)", value=False,
                                   help="Fetch fewer rows for faster testing")
        
        # Add button to fetch data
        fetch_button = st.sidebar.button("Fetch Data from BigQuery")
        
        # Fetch data either automatically or when button is pressed
        if fetch_button or (auto_fetch and 'data_source' not in st.session_state):
            # Store the current data source in session state to prevent auto-fetching on every rerun
            st.session_state['data_source'] = data_source
            
            try:
                with st.spinner("Fetching data from BigQuery..."):
                    df, refresh_time = fetch_expansion_data(limit_rows=test_mode)
                    if len(df) > 0:
                        # Add data diagnostic information
                        with st.expander("Data Diagnostics"):
                            st.subheader("Cities in dataset")
                            city_counts = df['city'].value_counts()
                            st.dataframe(city_counts)
                            
                            st.subheader("Data ranges")
                            st.write(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
                            st.write(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")
                            
                            # Check for any filtering that might be removing US cities
                            us_data = df[(df['latitude'] > 24) & (df['latitude'] < 50) & 
                                         (df['longitude'] > -125) & (df['longitude'] < -66)]
                            st.write(f"US data points (rough boundaries): {len(us_data)}")
                            if len(us_data) > 0:
                                us_cities = us_data['city'].value_counts()
                                st.dataframe(us_cities)
                            else:
                                st.warning("No US cities found in the data!")
                        
                        st.session_state['df'] = df
                        st.session_state['refresh_time'] = refresh_time
                        st.success(f"Successfully loaded {len(df)} records from BigQuery")
                        st.info(f"Data last refreshed: {refresh_time}")
                    else:
                        st.error("No data returned from query")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.code(traceback.format_exc())
        # If data is stored in session state, use it
        elif 'df' in st.session_state and 'refresh_time' in st.session_state:
            df = st.session_state['df']
            refresh_time = st.session_state['refresh_time']
            st.success(f"Using cached data: {len(df)} records")
            st.info(f"Data last refreshed: {refresh_time}")
    else:
        # Reset data source in session state when switching to CSV
        if 'data_source' in st.session_state and st.session_state['data_source'] != data_source:
            st.session_state['data_source'] = data_source
            if 'df' in st.session_state:
                del st.session_state['df']
            if 'refresh_time' in st.session_state:
                del st.session_state['refresh_time']
        
        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload your CSV file with delivery data", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data with more explicit parameters
                df = pd.read_csv(uploaded_file)
                
                # Store data in session state
                st.session_state['df'] = df
                
                # Show preview of the data
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head())
                
                # Debug information
                st.subheader("CSV Information")
                st.write(f"Columns found: {', '.join(df.columns.tolist())}")
                st.write(f"Number of rows: {len(df)}")
                
                # Check data types
                st.write("Data types:")
                st.write(df.dtypes)
                
                # Verify data has required columns
                required_columns = ['latitude', 'longitude', 'city', 'delivs_a', 'delivs_b', 'delivs_c']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please make sure your CSV contains all required columns: latitude, longitude, city, delivs_a, delivs_b, delivs_c")
                    return
                    
                st.success(f"Successfully loaded {len(df)} records from CSV")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                st.info("Please check that your CSV file is properly formatted")
                return
    
    # Continue only if data is loaded
    if not df.empty:
        # Verify data has required columns
        required_columns = ['latitude', 'longitude', 'city', 'delivs_a', 'delivs_b', 'delivs_c']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Data must contain these columns: {', '.join(required_columns)}")
            return
        
        # H3 Resolution slider - Added user control for cell size
        display_resolution = st.sidebar.slider(
            "H3 Cell Size (Resolution)",
            min_value=4,  # Larger cells
            max_value=9,  # Smaller cells
            value=6,      # Default value
            step=1,
            help="Lower values = larger hexagons, higher values = smaller hexagons"
        )
        
        # Base resolution (keeping this fixed but higher than display)
        base_resolution = 11
        
        # Transparency control
        transparency = st.sidebar.slider(
            "Cell Transparency",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Lower values = more transparent cells (better for seeing roads)"
        )
        
        # Updated color scheme options - making region-specific demand the default
        coloring_scheme = st.sidebar.radio(
            "Coloring Scheme",
            options=["Region-specific Demand", "Global Demand Scale", "Region-based Colors"],
            index=0,
            help="Region-specific: Colors cells based on demand within each region; Global: Colors based on demand across all regions; Region-based: Assigns distinct colors to each region"
        )
        
        # Determine coloring mode based on selection
        if coloring_scheme == "Region-specific Demand":
            use_demand_colors = True
            use_region_specific_scale = True
        elif coloring_scheme == "Global Demand Scale":
            use_demand_colors = True
            use_region_specific_scale = False
        else:  # Region-based Colors
            use_demand_colors = False
            use_region_specific_scale = False
        
        # Census tract data toggle
        show_census = st.sidebar.checkbox("Show census tract data", value=False)
        census_tract_data = None  # You would need to load this if show_census is True
        
        # Add download options
        with st.sidebar.expander("Advanced Options"):
            # Maximum cells to render
            max_cells = st.slider(
                "Maximum cells to render",
                min_value=1000,
                max_value=20000,
                value=10000,
                step=1000,
                help="Higher values will show more detail but may slow down the browser"
            )
            
            # Option to show/hide tooltips on hover
            show_tooltips = st.checkbox("Show tooltips on hover", value=True)
        
        # Create map button
        if st.sidebar.button("Generate Map"):
            try:
                with st.spinner("Creating H3 map..."):
                    # Add more detailed status messages
                    status_container = st.empty()
                    status_container.info("Step 1: Processing data...")
                    
                    # Add more verbose checking of data
                    if df.empty:
                        st.error("DataFrame is empty. Please check your data source.")
                        return
                    
                    # Check if coordinate columns contain valid numerical data
                    try:
                        # Explicitly convert coordinate columns to float
                        df['latitude'] = pd.to_numeric(df['latitude'], errors='raise')
                        df['longitude'] = pd.to_numeric(df['longitude'], errors='raise')
                        df['delivs_a'] = pd.to_numeric(df['delivs_a'], errors='coerce').fillna(0)
                        df['delivs_b'] = pd.to_numeric(df['delivs_b'], errors='coerce').fillna(0)
                        df['delivs_c'] = pd.to_numeric(df['delivs_c'], errors='coerce').fillna(0)
                        
                        # Check for invalid coordinates and filter out bad data
                        before_filter = len(df)
                        df = df[
                            (df['latitude'].between(-90, 90)) & 
                            (df['longitude'].between(-180, 180))
                        ]
                        after_filter = len(df)
                        
                        if before_filter > after_filter:
                            st.warning(f"Filtered out {before_filter - after_filter} rows with invalid coordinates.")
                        
                        # Verify we have both US and international data
                        us_data = df[(df['latitude'] > 24) & (df['latitude'] < 50) & 
                                     (df['longitude'] > -125) & (df['longitude'] < -66)]
                        
                        intl_data = df[~((df['latitude'] > 24) & (df['latitude'] < 50) & 
                                       (df['longitude'] > -125) & (df['longitude'] < -66))]
                        
                        st.info(f"Data contains {len(us_data)} US points and {len(intl_data)} international points")
                        
                        # Check for NaN values after conversion
                        if df['latitude'].isna().any() or df['longitude'].isna().any():
                            st.error("Some latitude/longitude values could not be converted to numbers. Please check your data.")
                            return
                    
                    except Exception as e:
                        st.error(f"Error processing coordinates: {str(e)}")
                        st.code(traceback.format_exc())
                        return
                    
                    # Ensure 'city' column contains strings
                    if 'city' in df.columns:
                        df['city'] = df['city'].astype(str)
                    
                    # Update status
                    status_container.info("Step 2: Creating H3 cells...")
                    
                    # Try to create the map with additional error handling
                    try:
                        # Data sampling section - modified to preserve regional representation
                        if len(df) > 25000:
                            # Get unique regions/cities before sampling to ensure representation
                            regions = df['city'].unique()
                            st.info(f"Large dataset detected ({len(df)} rows) with {len(regions)} unique regions.")
                            
                            # Option to disable sampling
                            disable_sampling = st.checkbox("Disable sampling (may affect performance)", value=False)
                            
                            if not disable_sampling:
                                # Stratified sampling to ensure all regions are represented
                                sampled_df = pd.DataFrame()
                                for region in regions:
                                    region_df = df[df['city'] == region]
                                    # Calculate sample size proportionally but ensure minimum representation
                                    sample_size = max(100, int(len(region_df) * 20000 / len(df)))
                                    # If region is smaller than desired sample, take all rows
                                    if len(region_df) <= sample_size:
                                        sampled_region = region_df
                                    else:
                                        sampled_region = region_df.sample(sample_size, random_state=42)
                                    sampled_df = pd.concat([sampled_df, sampled_region])
                                
                                st.success(f"Sampled {len(sampled_df)} rows while preserving all {len(regions)} regions.")
                                df = sampled_df
                        
                        m, cells = create_hierarchical_h3_map(
                            df=df,
                            display_resolution=display_resolution,
                            base_resolution=base_resolution,
                            use_demand_colors=use_demand_colors,
                            use_region_specific_scale=use_region_specific_scale,
                            census_tract_data=census_tract_data,
                            transparency_level=transparency,
                            max_cells=max_cells,
                            show_tooltips=show_tooltips
                        )
                        
                        # Update status
                        status_container.info("Step 3: Rendering map...")
                        
                        # Display some stats about the map
                        if cells:
                            st.write(f"Created {len(cells)} H3 cells at resolution {display_resolution}")
                        else:
                            st.warning("No H3 cells were created. This may indicate an issue with the data.")
                        
                        # Add container for the map with expanded width
                        map_container = st.container()
                        with map_container:
                            # Display the map with more specific parameters
                            if m is not None:
                                st_folium(m, width=1200, height=700, returned_objects=[])
                            else:
                                st.error("Failed to create map. Please check the logs for errors.")
                        
                        # Clear the status message
                        status_container.empty()
                        
                    except Exception as e:
                        st.error(f"Error creating map: {str(e)}")
                        st.code(traceback.format_exc())
                        return
                    
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.code(traceback.format_exc())
    else:
        if data_source == "BigQuery Database":
            st.info("Click 'Fetch Data from BigQuery' in the sidebar to load data")
        else:
            st.info("Please upload a CSV file to begin")
            
            # Only show example data format for CSV option
            if data_source == "CSV Upload":
                st.markdown("### CSV should contain these required columns:")
                st.code("latitude, longitude, city, delivs_a, delivs_b, delivs_c")
                
                # Provide a sample CSV structure
                st.markdown("### Sample CSV Format:")
                sample_data = """
                latitude,longitude,city,delivs_a,delivs_b,delivs_c
                37.7749,-122.4194,San Francisco,120.5,85.2,45.1
                34.0522,-118.2437,Los Angeles,95.3,62.8,31.4
                40.7128,-74.0060,New York,150.2,105.7,68.9
                """
                st.code(sample_data)
                
                # Add download button for sample CSV
                import base64
                csv_string = """latitude,longitude,city,delivs_a,delivs_b,delivs_c
37.7749,-122.4194,San Francisco,120.5,85.2,45.1
34.0522,-118.2437,Los Angeles,95.3,62.8,31.4
40.7128,-74.0060,New York,150.2,105.7,68.9
32.7157,-117.1611,San Diego,80.7,55.3,29.8
47.6062,-122.3321,Seattle,110.3,75.9,40.2
41.8781,-87.6298,Chicago,130.6,89.4,48.7
39.9526,-75.1652,Philadelphia,85.4,59.1,32.6
42.3601,-71.0589,Boston,95.8,66.2,35.9
29.7604,-95.3698,Houston,105.2,72.5,39.3
33.4484,-112.0740,Phoenix,75.9,52.4,28.1"""
                
                b64 = base64.b64encode(csv_string.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sample_delivery_data.csv">Download Sample CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

# Add a footer with information
def footer():
    st.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-top:20px;">
        <p style="text-align:center;margin:0;">
            H3 Delivery Map - Built with Streamlit and H3 Hexagonal Hierarchical Geospatial Indexing System
        </p>
        <p style="text-align:center;font-size:0.8em;margin-top:5px;">
            Version 1.2 - Last updated: April 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()