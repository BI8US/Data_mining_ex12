import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.express as px
import plotly.graph_objects as go
# Removed unused: from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from datetime import date

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Energy Consumption Analysis")

st.title("Electricity Consumption Patterns Visualizer")

# --- Data Loading Function with Caching ---
@st.cache_data
def load_and_process_data():
    api_url = "https://decision.cs.taltech.ee/electricity/api/"
    base_data_url = "https://decision.cs.taltech.ee/electricity/data/"
    my_dataset_hash = "6df7cad937de1f461747833479"  # Your dataset hash

    st.info("Loading dataset catalog from API...")
    try:
        resp = requests.get(api_url)
        resp.raise_for_status()
        catalog = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to TalTech API: {e}. Please check your internet connection or API availability.")
        return {}, {}

    datasets_to_process_info = []
    added_hashes = set()

    # Add your dataset first
    if my_dataset_hash not in added_hashes:
        datasets_to_process_info.append({"dataset": my_dataset_hash})
        added_hashes.add(my_dataset_hash)

    # Add 5 additional unique datasets from the catalog
    for entry in catalog:
        if entry['dataset'] not in added_hashes:
            datasets_to_process_info.append(entry)
            added_hashes.add(entry['dataset'])
        if len(datasets_to_process_info) >= 6:  # Ensure 5 additional + your dataset
            break

    if len(datasets_to_process_info) == 0:
        st.error("No datasets found to process. Please check API connection or dataset hash.")
        return {}, {}

    all_datasets = {}
    dataset_summaries = {}  # New dictionary to store summaries for sidebar
    st.info("Downloading and preprocessing selected CSV files...")
    progress_bar = st.progress(0)
    for i, entry_info in enumerate(datasets_to_process_info):
        hash_id = entry_info['dataset']
        csv_url = f"{base_data_url}{hash_id}.csv"

        try:
            resp_csv = requests.get(csv_url)
            resp_csv.raise_for_status()
            txt = resp_csv.content.decode('latin1')

            lines = txt.splitlines()
            header_idx = next((j for j, l in enumerate(lines) if l.startswith('Periood;')), None)
            if header_idx is None:
                st.warning(f"Header not found in {hash_id}.csv, skipping.")
                continue

            header_line = lines[header_idx].strip().rstrip(',').strip()
            data_block = header_line + "\n" + "\n".join(lines[header_idx + 1:])

            df = pd.read_csv(io.StringIO(data_block), sep=';', decimal=',', usecols=[0, 1])
            df.columns = [col.strip() for col in df.columns]

            new_column_names = {}
            datetime_col_found = None
            consumption_col_found = None

            for col in df.columns:
                if 'periood' in col.lower():
                    datetime_col_found = col
                    new_column_names[col] = 'Datetime'
                elif 'energia' in col.lower() or 'kwh' in col.lower():
                    consumption_col_found = col
                    new_column_names[col] = 'Consumption'

            if datetime_col_found is None or consumption_col_found is None:
                st.warning(
                    f"Could not find 'Periood' or 'Consumption' equivalent column in dataset {hash_id}. Skipping.")
                continue

            df = df.rename(columns=new_column_names)
            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M', dayfirst=True, errors='coerce')
            df.dropna(subset=['Datetime', 'Consumption'], inplace=True)
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')
            df.dropna(subset=['Consumption'], inplace=True)

            if not df.empty:
                all_datasets[hash_id] = df
                # Store summary info for sidebar display
                min_date_str = df.index.min().strftime('%Y-%m-%d')
                max_date_str = df.index.max().strftime('%Y-%m-%d')
                total_days = (df.index.max() - df.index.min()).days + 1
                dataset_summaries[hash_id] = {
                    'min_date': min_date_str,
                    'max_date': max_date_str,
                    'total_days': total_days,
                }
            else:
                st.warning(f"Dataset {hash_id} is empty after preprocessing. Skipping.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading {csv_url}: {e}")
        except Exception as e:
            st.error(f"Error processing {hash_id}.csv: {e}")

        progress_bar.progress((i + 1) / len(datasets_to_process_info))

    if all_datasets:
        st.success("Data successfully loaded and preprocessed!")
    return all_datasets, dataset_summaries


# --- Load data on app start ---
all_datasets, dataset_summaries = load_and_process_data()
original_hashes = list(all_datasets.keys())  # Get the list of all dataset hashes

# Check if any datasets were loaded
if not all_datasets:
    st.error("No datasets could be loaded. The application cannot proceed.")
    st.stop()  # Stop Streamlit execution if no data is available

# --- Sidebar for Dataset Selection (Button + Info - Compact) ---
st.sidebar.header("Select Dataset for Analysis")

# Initialize session state for selected dataset if not already set
if 'selected_dataset_hash_sidebar' not in st.session_state:
    if original_hashes:
        st.session_state.selected_dataset_hash_sidebar = original_hashes[0]
    else:
        st.session_state.selected_dataset_hash_sidebar = None

if not dataset_summaries:
    st.sidebar.warning("No datasets to display.")
    st.stop()

# Inject CSS for compact dataset displays (kept as is due to complexity and user preference)
st.markdown(
    """
    <style>
    /* Reduce overall vertical spacing in the sidebar */
    div[data-testid="stSidebarContent"] div.stVerticalBlock {
        gap: 0.3rem; /* Reduced gap between blocks even more */
    }

    /* Style for the actual Streamlit button */
    div[data-testid="stSidebarContent"] div.stButton > button {
        background-color: #262730;
        color: White;
        border: 1px solid #ccc; /* Light border */
        padding: 0.2em 0.5em; /* FURTHER REDUCED PADDING */
        margin-bottom: 0.05rem; /* VERY SMALL margin below button to info */
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.8em; /* SMALLER FONT SIZE */
        width: 100%;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease; /* Added color transition */
    }
    /* Hover effect for buttons */
    div[data-testid="stSidebarContent"] div.stButton > button:hover:not([aria-disabled="true"]) {
        background-color: #FF4B4B;
        color: White; /* Ensure text is white on hover */
        border-color: #999;
    }
    /* Style for the active/pressed state of the button */
    div[data-testid="stSidebarContent"] div.stButton > button:active {
        color: White !important; /* Ensure text is white on active */
        background-color: #367C39; /* Slightly darker green when active */
        border-color: #367C39;
    }
    /* Style for the focused state of the button (e.g., via Tab key) */
    div[data-testid="stSidebarContent"] div.stButton > button:focus {
        outline: none; /* Remove default blue/red focus outline */
        box_shadow: 0 0 0 0.15rem rgba(76, 175, 80, 0.5); /* Add custom green focus outline */
        color: White !important; /* Ensure text is white on focus */
    }
    /* Style for the selected (disabled) button */
    div[data-testid="stSidebarContent"] div.stButton > button[aria-disabled="true"] {
        background-color: #4CAF50 !important; /* Green background for selected button */
        color: white !important; /* White text for selected button */
        border-color: #4CAF50 !important; /* Green border for selected button */
        cursor: default !important; /* Default cursor for selected button */
    }

    /* Style for the text info paragraphs directly following a button */
    div[data-testid="stSidebarContent"] div.stButton + div > p {
        margin-top: 0 !important; /* No margin above first info line */
        margin-bottom: 0 !important; /* No margin between info lines */
        font-size: 0.75em !important; /* EVEN SMALLER FONT SIZE for info */
        line-height: 1.1 !important; /* EVEN TIGHTER line spacing */
        padding-left: 0.5em; /* Align with button text padding */
    }

    /* Target the container of the selected button (which is a div.stButton)
       and style its border and background to wrap the button AND the following text.
       This is the tricky part, as we need to target sibling elements.
    */
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) {
        border: 2px solid #4CAF50; /* Green border for the selected button's parent div */
        background-color: rgba(76, 175, 80, 0.1); /* Light green background */
        border-radius: 0.25rem;
        padding: 0.25rem 0.5rem; /* Match outer padding of previous container approach */
        margin-bottom: 0.5rem; /* Space after the entire selected block */
    }

    /* Adjust the margins of paragraphs *within* the selected block */
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) + div > p {
        /* This rule is now inside the more general rule above for p tags, and applies here too */
    }

    /* General styling for divs that contain st.markdown, to reduce their padding */
    div[data-testid="stSidebarContent"] div.stMarkdown {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Ensure text within the selected block is aligned correctly */
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) ~ div.stMarkdown > p {
        padding-left: 0.5em; /* Re-apply left padding for consistency if needed */
        font-size: 0.75em; /* Keep font size consistent with other info */
    }

    /* Target the text below a selected button specifically to apply the selected background/border
       This is the hardest part. The previous selector was not correct.
       We need to select the siblings of the selected button's parent div.
       Streamlit wraps `st.button` in a `div.stButton` and `st.markdown` in a `div.stMarkdown`.
       So we need to select all `div.stMarkdown` that are siblings of the selected `div.stButton`.
    */
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) + div.stMarkdown,
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) + div.stMarkdown + div.stMarkdown,
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) + div.stMarkdown + div.stMarkdown + div.stMarkdown,
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) + div.stMarkdown + div.stMarkdown + div.stMarkdown + div.stMarkdown
    {
        background-color: rgba(76, 175, 80, 0.1); /* Match selected background */
        border-left: 2px solid #4CAF50; /* Match selected border */
        border-right: 2px solid #4CAF50;
        padding: 0.25rem 0.5rem; /* Match outer padding of button's parent */
        /* To make it appear as a single block, these must align with the button's parent. */
        margin-left: -0.5rem !important; /* Compensate for padding of the button's parent */
        margin-right: -0.5rem !important; /* Compensate for padding of the button's parent */
        padding-left: calc(0.5rem + 2px) !important; /* Adjust for border and parent padding */
        padding-right: 0.5rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Specific styling for the last element in the group to get bottom border and rounded corners */
    div[data-testid="stSidebarContent"] div.stButton:has(button[aria-disabled="true"]) + div.stMarkdown + div.stMarkdown + div.stMarkdown + div.stMarkdown {
        border-bottom: 2px solid #4CAF50;
        border-bottom-left-radius: 0.25rem;
        border-bottom-right-radius: 0.25rem;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

for hash_id, summary in dataset_summaries.items():
    is_selected = (st.session_state.selected_dataset_hash_sidebar == hash_id)

    if st.sidebar.button(f"{hash_id}", key=f"select_ds_btn_{hash_id}", use_container_width=True,
                         disabled=is_selected):
        st.session_state.selected_dataset_hash_sidebar = hash_id
        st.rerun()

    st.sidebar.markdown(f"<p style='margin-bottom: 0.2rem;'><b>Start:</b> {summary['min_date']} <b>End:</b> {summary['max_date']}</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p><b>Days:</b> {summary['total_days']}</p>", unsafe_allow_html=True)


# Update the main selected_dataset_hash from session state
selected_dataset_hash = st.session_state.selected_dataset_hash_sidebar

if selected_dataset_hash is None:
    st.error("Error: No dataset selected or available. Please ensure datasets are loaded correctly.")
    st.stop()

# Set the current DataFrame based on the selected hash
current_df = all_datasets[selected_dataset_hash]

# --- Section 1: Visualize One Dataset (Task 1) ---
st.header("1. Visualize One Dataset")
st.subheader(f"Dataset **{selected_dataset_hash}**")
st.info("The default visualization period is set to the last 100 days of available data.")

# Determine the available date range for date pickers
max_date = current_df.index.max().date()
min_date = current_df.index.min().date()

# Suggest a 100-day period ending at max_date, or shorter if data is less than 100 days
start_date_initial = max_date - pd.Timedelta(days=99)
if start_date_initial < min_date:
    start_date_initial = min_date

col1_date_picker, col2_date_picker = st.columns(2)
with col1_date_picker:
    start_viz_date = st.date_input(
        "Start date for visualization:",
        value=start_date_initial,
        min_value=min_date,
        max_value=max_date
    )
with col2_date_picker:
    end_viz_date = st.date_input(
        "End date for visualization:",
        value=max_date,
        min_value=start_viz_date,  # End date cannot be before start date
        max_value=max_date
    )

# Filter data for the selected range
df_selected_period = current_df[(current_df.index.date >= start_viz_date) & (current_df.index.date <= end_viz_date)]

if df_selected_period.empty:
    st.warning("No data available for the selected date range. Please choose a different range.")
else:
    # --- Prepare data for pattern visualization (moved outside if/elif blocks) ---
    df_processed_for_patterns = df_selected_period.copy()
    df_processed_for_patterns['Date'] = df_processed_for_patterns.index.date
    df_processed_for_patterns['Hour'] = df_processed_for_patterns.index.hour
    df_processed_for_patterns['DayOfWeek'] = df_processed_for_patterns.index.day_name()

    # Filter for full days (24 entries) to ensure accurate daily profiles
    daily_record_counts = df_processed_for_patterns['Date'].value_counts()
    full_days_in_period = daily_record_counts[daily_record_counts == 24].index
    df_full_days_in_period = df_processed_for_patterns[df_processed_for_patterns['Date'].isin(full_days_in_period)].copy()

    # Create pivot table for heatmap (if full days exist)
    df_daily_consumption_pivot = pd.DataFrame()
    if not df_full_days_in_period.empty:
        df_daily_consumption_pivot = df_full_days_in_period.pivot_table(index='Date', columns='Hour',
                                                                        values='Consumption')

    viz_type_single_dataset = st.radio(
        "Choose visualization type:",
        ('Full Period Consumption Line', 'Heatmap', 'Average Daily Profile (with variability)',
         'Average Consumption by Day of Week'),
        key='single_dataset_viz_type'
    )

    if viz_type_single_dataset == 'Full Period Consumption Line':
        st.subheader("Consumption Line Over Selected Period")

        resolution_option = st.radio(
            "Select time resolution:",
            ('Hourly', 'Daily', 'Weekly'),
            key='resolution_full_line'
        )

        df_temp = df_selected_period.copy()  # Работаем с копией, чтобы не изменять оригинал

        # Ensure Datetime is a column at the very beginning if it's currently the index
        if df_temp.index.name == 'Datetime':
            df_temp = df_temp.reset_index()

        # Now, df_temp should have 'Datetime' and 'Consumption' as columns

        if resolution_option == 'Daily':
            # Resample and sum 'Consumption', then reset index to get 'Datetime' column back
            df_to_plot = df_temp.set_index('Datetime')['Consumption'].resample('D').sum().reset_index()
        elif resolution_option == 'Weekly':
            # Resample and sum 'Consumption', then reset index to get 'Datetime' column back
            df_to_plot = df_temp.set_index('Datetime')['Consumption'].resample('W').sum().reset_index()
        else:  # 'Hourly'
            # For hourly, df_to_plot is already in the correct format (Datetime and Consumption as columns)
            # if the initial reset_index() was performed.
            df_to_plot = df_temp.copy()  # Make sure to get a fresh copy of the processed df_temp
            # No need for resample, just ensure column names are correct
            df_to_plot = df_to_plot[['Datetime', 'Consumption']]  # Select only relevant columns

        fig_line_full = px.line(
            df_to_plot,
            x='Datetime',
            y='Consumption',
            title=f'Consumption Trend from {start_viz_date} to {end_viz_date} ({resolution_option} Resolution)',
            labels={'Datetime': f'{resolution_option} Start Date', 'Consumption': 'Consumption (kWh)'},
            height=500
        )
        fig_line_full.update_layout(hovermode="x unified")
        st.plotly_chart(fig_line_full, use_container_width=True)

    elif viz_type_single_dataset == 'Heatmap':
        if df_daily_consumption_pivot.empty:
            st.warning("No complete days (24 hours of records) in the selected range to generate a heatmap.")
        else:
            st.subheader("Hourly Consumption Heatmap")
            fig_heatmap = px.imshow(
                df_daily_consumption_pivot.T,
                labels=dict(x="Date", y="Hour of Day", color="Consumption (kWh)"),
                x=[d.strftime('%Y-%m-%d') for d in df_daily_consumption_pivot.index],
                y=df_daily_consumption_pivot.columns,
                color_continuous_scale="Viridis",
                title=f'Hourly Consumption Heatmap ({start_viz_date} - {end_viz_date})',
                height=700
            )
            fig_heatmap.update_xaxes(side="top")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    elif viz_type_single_dataset == 'Average Daily Profile (with variability)':
        if df_full_days_in_period.empty:
            st.warning(
                "No complete days (24 hours of records) in the selected range to calculate average daily profile.")
        else:
            st.subheader("Average Daily Consumption Profile")
            # Calculate mean, std, min, and max for each hour
            hourly_stats = df_full_days_in_period.groupby('Hour')['Consumption'].agg(['mean', 'std', 'min', 'max']).reset_index()

            fig_avg_profile = go.Figure()

            # Add min line
            fig_avg_profile.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['min'],
                mode='lines',
                name='Minimum Consumption',
                line=dict(color='lightgreen', width=1, dash='dot')
            ))

            # Add max line
            fig_avg_profile.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['max'],
                mode='lines',
                name='Maximum Consumption',
                line=dict(color='orange', width=1, dash='dot')
            ))

            # Add mean line
            fig_avg_profile.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['mean'],
                mode='lines',
                name='Average Consumption',
                line=dict(color='skyblue', width=3)
            ))

            # Add shaded area for variability (mean +/- std)
            fig_avg_profile.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['mean'] + hourly_stats['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_avg_profile.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['mean'] - hourly_stats['std'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,150,200,0.3)', # Increased transparency
                name='Mean ± Std Dev',
                line=dict(width=0)
            ))

            fig_avg_profile.update_layout(
                title=f'Average, Min/Max Daily Consumption Profile ({start_viz_date} - {end_viz_date})',
                xaxis_title='Hour of Day',
                yaxis_title='Consumption (kWh)',
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig_avg_profile, use_container_width=True)

    elif viz_type_single_dataset == 'Average Consumption by Day of Week':
        if df_full_days_in_period.empty:
            st.warning(
                "No complete days (24 hours of records) in the selected range to generate average consumption by day of week.")
        else:
            st.subheader("Average Consumption by Day of Week")
            daily_avg_by_dayofweek = df_full_days_in_period.groupby(['DayOfWeek', 'Hour'])[
                'Consumption'].mean().unstack(level=0)

            # Reorder days for consistent plotting (Monday to Sunday)
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_avg_by_dayofweek = daily_avg_by_dayofweek.reindex(columns=ordered_days, fill_value=np.nan).dropna(
                axis=1, how='all')

            if not daily_avg_by_dayofweek.empty:
                fig_weekday_profiles = px.line(
                    daily_avg_by_dayofweek,
                    x=daily_avg_by_dayofweek.index,
                    y=daily_avg_by_dayofweek.columns,
                    title='Average Hourly Consumption by Day of Week',
                    labels={'value': 'Average Consumption (kWh)', 'index': 'Hour of Day', 'DayOfWeek': 'Day of Week'},
                    height=500
                )
                fig_weekday_profiles.update_layout(hovermode="x unified", legend_title_text='Day of Week')
                st.plotly_chart(fig_weekday_profiles, use_container_width=True)
            else:
                st.warning("Not enough data to generate average consumption by day of week.")

st.markdown("---")

# --- Section 2: Compare Daily Profiles of Multiple Consumers for a Single Day (Task 2) ---
st.header("2. Compare Daily Profiles of Multiple Consumers for a Single Day")

# Get a list of all available dates across all datasets
all_dates = pd.Index([])
for df_hash in all_datasets:
    all_dates = all_dates.union(all_datasets[df_hash].index.normalize())

if all_dates.empty:
    st.warning("No available dates to compare consumers.")
else:
    # Select a date for analysis
    max_compare_date = all_dates.max()
    min_compare_date = all_dates.min()

    selected_compare_date = st.date_input(
        "Select a date to compare consumers:",
        value=min(max_compare_date.date(), date.today()),
        min_value=min_compare_date.date(),
        max_value=max_compare_date.date()
    )

    # Prepare data for comparison
    compare_data = []
    available_hashes_for_date = []

    for hash_id, df_consumer in all_datasets.items():
        df_day = df_consumer[(df_consumer.index.date == selected_compare_date)]
        if len(df_day) == 24:
            df_day = df_day.copy()
            df_day['Hour'] = df_day.index.hour
            df_day['Consumer'] = hash_id
            compare_data.append(df_day)
            available_hashes_for_date.append(hash_id)

    if not compare_data:
        st.warning(
            f"No data for {selected_compare_date} available for consumer comparison (or no complete 24-hour profiles).")
    else:
        df_all_consumers_selected_day = pd.concat(compare_data)

        viz_type_multi_dataset = st.radio(
            "Choose visualization type:",
            ('Overlay Consumer Plots', 'Cluster Consumers'),
            key='multi_dataset_viz_type'
        )

        if viz_type_multi_dataset == 'Overlay Consumer Plots':
            st.subheader(f"Daily Profiles of All Available Consumers for {selected_compare_date}")
            fig_multi_consumer = px.line(
                df_all_consumers_selected_day,
                x='Hour',
                y='Consumption',
                color='Consumer',
                title=f'Daily Consumption Profiles for Different Consumers on {selected_compare_date}',
                labels={'Consumption': 'Consumption (kWh)', 'Hour': 'Hour of Day', 'Consumer': 'Consumer ID'},
                height=600
            )
            fig_multi_consumer.update_traces(opacity=0.7)
            fig_multi_consumer.update_layout(hovermode="x unified", legend_title_text='Consumer ID')
            st.plotly_chart(fig_multi_consumer, use_container_width=True)

        elif viz_type_multi_dataset == 'Cluster Consumers':
            st.subheader(f"Clustering of Consumer Daily Profiles for {selected_compare_date}")

            # Prepare data for clustering: consumers as rows, hours as columns
            df_pivot_consumers = df_all_consumers_selected_day.pivot_table(index='Consumer', columns='Hour',
                                                                           values='Consumption')

            num_available_consumers = len(df_pivot_consumers)

            if num_available_consumers < 2:
                st.warning(f"Not enough consumers ({num_available_consumers}) with complete daily profiles for {selected_compare_date} to perform clustering. Need at least 2 consumers.")
            elif num_available_consumers == 2:
                # If exactly 2 consumers, clustering must be into 2 clusters
                num_clusters = 2
                st.info(f"Only 2 consumers available. Performing clustering into {num_clusters} clusters.")
            else:
                # If more than 2 consumers, allow selection via slider
                # max_clusters_allowed is minimum of actual consumers or 5 (as per original logic)
                max_clusters_allowed = min(num_available_consumers, 5)
                # Default value should also be valid, typically start with 3 or max_clusters_allowed if it's less than 3
                initial_slider_value = min(3, max_clusters_allowed)

                num_clusters = st.slider(
                    "Select number of clusters (K):",
                    min_value=2,
                    max_value=max_clusters_allowed,
                    value=initial_slider_value
                )

            # Continue with clustering only if num_clusters is determined (i.e., num_available_consumers >= 2)
            if num_available_consumers >= 2:
                try:
                    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                    kmeans_model.fit(df_pivot_consumers)
                    cluster_labels = kmeans_model.labels_
                    cluster_centroids = kmeans_model.cluster_centers_

                    # Add cluster labels to the pivoted DataFrame for easy access
                    df_pivot_consumers['Cluster'] = cluster_labels

                    # Visualize centroids
                    fig_centroids = go.Figure()
                    for i in range(num_clusters):
                        fig_centroids.add_trace(go.Scatter(
                            x=df_pivot_consumers.columns[:-1],  # Hours
                            y=cluster_centroids[i],
                            mode='lines',
                            name=f'Cluster {i} (N={np.sum(cluster_labels == i)})',
                            line=dict(width=3)
                        ))
                    fig_centroids.update_layout(
                        title=f'Energy Consumption Cluster Centroids for {selected_compare_date}',
                        xaxis_title='Hour of Day',
                        yaxis_title='Average Consumption (kWh)',
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_centroids, use_container_width=True)

                    st.subheader("Distribution of Consumers by Cluster")
                    cluster_counts = df_pivot_consumers['Cluster'].value_counts().sort_index()
                    st.dataframe(cluster_counts.rename("Number of Consumers").to_frame())

                    # Optional: show each consumer and its assigned cluster
                    st.write("Consumers and their assigned clusters:")
                    st.dataframe(df_pivot_consumers[['Cluster']])

                except Exception as e:
                    st.error(
                        f"Error during clustering: {e}. Please ensure there is enough data and it is correctly formatted.")
