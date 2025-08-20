"""
Complete Drought Monitoring System using Standard Precipitation Index (SPI)
Author: AI Assistant
Version: 1.0
Description: Real-time drought monitoring with working APIs and proper error handling
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import warnings
import requests
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_scalar_value(value):
    """
    Safely convert numpy arrays or other types to scalar values
    
    Args:
        value: Any numeric value, array, or pandas object
        
    Returns:
        float: Scalar value or NaN if conversion fails
    """
    try:
        if pd.isna(value):
            return np.nan
        
        if hasattr(value, 'item'):
            return float(value.item())
        elif isinstance(value, (list, np.ndarray)):
            if len(value) > 0:
                return float(value[0]) if not pd.isna(value[0]) else np.nan
            else:
                return np.nan
        else:
            return float(value) if not pd.isna(value) else np.nan
    except:
        return np.nan

def format_date(date_obj, format_str='%Y-%m-%d'):
    """Format datetime object to string"""
    try:
        if isinstance(date_obj, pd.Timestamp):
            return date_obj.strftime(format_str)
        elif isinstance(date_obj, datetime):
            return date_obj.strftime(format_str)
        return str(date_obj)
    except:
        return "Invalid Date"

def get_color_scale(spi_value):
    """Get color based on SPI value for visualization"""
    spi_val = safe_scalar_value(spi_value)
    
    if pd.isna(spi_val):
        return '#CCCCCC'  # Gray for missing data
    elif spi_val >= 2.0:
        return '#000080'  # Dark blue - Extremely wet
    elif spi_val >= 1.5:
        return '#0000FF'  # Blue - Very wet
    elif spi_val >= 1.0:
        return '#87CEEB'  # Sky blue - Moderately wet
    elif spi_val >= -1.0:
        return '#90EE90'  # Light green - Near normal
    elif spi_val >= -1.5:
        return '#FFFF00'  # Yellow - Moderately dry
    elif spi_val >= -2.0:
        return '#FFA500'  # Orange - Severely dry
    else:
        return '#FF0000'  # Red - Extremely dry

def create_test_dataset(lat, lon, years_back):
    """
    Create a test dataset with realistic SPI patterns for debugging
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        years_back (int): Number of years of data to generate
        
    Returns:
        dict: Dataset with synthetic precipitation data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    # Create monthly date range
    monthly_dates = pd.date_range(start_date, end_date, freq='M')
    
    # Generate realistic precipitation with clear patterns
    np.random.seed(42)  # For reproducible results
    precip_values = []
    
    for i, date in enumerate(monthly_dates):
        month = date.month
        year = date.year
        
        # Indonesian monsoon pattern
        if month in [11, 12, 1, 2, 3, 4]:  # Wet season
            base_precip = np.random.gamma(3, 80)  # Higher precipitation
        else:  # Dry season
            base_precip = np.random.gamma(1.5, 20)  # Lower precipitation
        
        # Add some drought years (every 7-10 years)
        if (year - 2010) % 8 == 0:  # Drought years
            base_precip *= 0.3
        
        # Add some very wet years
        if (year - 2012) % 6 == 0:  # Wet years
            base_precip *= 2.0
        
        # Add trend and variability
        trend = 0.02 * i  # Slight upward trend
        noise = np.random.normal(0, base_precip * 0.1)
        
        final_precip = max(0, base_precip + trend + noise)
        precip_values.append(final_precip)
    
    # Create DataFrame
    df = pd.DataFrame({
        'precipitation': precip_values
    }, index=monthly_dates)
    
    return {
        'source': 'Test Dataset (Synthetic Indonesian Climate)',
        'data': df,
        'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'note': 'Synthetic data with realistic Indonesian climate patterns including droughts and wet periods',
        'monthly_points': len(df),
        'features': ['Monsoon patterns', 'Drought cycles', 'Wet periods', 'Seasonal variation']
    }

# ============================================================================
# DATA COLLECTION CLASS
# ============================================================================

class IntegratedClimateDataCollector:
    """Integrated climate data collector with working APIs tested in Postman"""
    
    def __init__(self):
        self.apis = {
            'open_meteo_archive': {
                'base_url': 'https://archive-api.open-meteo.com/v1/archive',
                'description': 'Historical weather data from 1940-2022'
            },
            'open_meteo_forecast': {
                'base_url': 'https://api.open-meteo.com/v1/forecast', 
                'description': 'Current and forecast data with 7-day history'
            },
            'era5_reanalysis': {
                'base_url': 'https://archive-api.open-meteo.com/v1/era5',
                'description': 'ERA5 reanalysis data from 1940-present'
            }
        }
    
    def get_open_meteo_data(self, lat, lon, start_date, end_date):
        """Get historical data using working Open-Meteo Archive API"""
        url = self.apis['open_meteo_archive']['base_url']
        
        # Convert dates to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'precipitation_sum,temperature_2m_mean',
            'timezone': 'auto'
        }
        
        try:
            st.info(f"🌐 Fetching data from Open-Meteo Archive API...")
            st.write(f"📍 Coordinates: {lat}, {lon}")
            st.write(f"📅 Period: {start_date} to {end_date}")
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Data successfully retrieved from Open-Meteo!")
                
                # Convert to pandas DataFrame
                if 'daily' in data and 'time' in data['daily']:
                    dates = pd.to_datetime(data['daily']['time'])
                    precip = data['daily']['precipitation_sum']
                    
                    df = pd.DataFrame({
                        'date': dates,
                        'precipitation': precip
                    })
                    df.set_index('date', inplace=True)
                    
                    # Handle missing values
                    df['precipitation'] = pd.to_numeric(df['precipitation'], errors='coerce').fillna(0)
                    
                    return {
                        'source': 'Open-Meteo Archive API',
                        'data': df,
                        'period': f"{start_date} to {end_date}",
                        'api_url': url,
                        'data_points': len(df)
                    }
                else:
                    st.error("❌ No daily data found in API response")
                    return None
            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"❌ Exception occurred: {e}")
            return None
    
    def get_era5_data(self, lat, lon, start_date, end_date):
        """Get ERA5 reanalysis data as backup"""
        url = self.apis['era5_reanalysis']['base_url']
        
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'precipitation_sum',
            'timezone': 'auto'
        }
        
        try:
            st.info("🌐 Trying ERA5 Reanalysis API as backup...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Data retrieved from ERA5!")
                
                if 'daily' in data and 'time' in data['daily']:
                    dates = pd.to_datetime(data['daily']['time'])
                    precip = data['daily']['precipitation_sum']
                    
                    df = pd.DataFrame({
                        'date': dates,
                        'precipitation': precip
                    })
                    df.set_index('date', inplace=True)
                    df['precipitation'] = pd.to_numeric(df['precipitation'], errors='coerce').fillna(0)
                    
                    return {
                        'source': 'ERA5 Reanalysis API',
                        'data': df,
                        'period': f"{start_date} to {end_date}",
                        'api_url': url
                    }
            return None
            
        except Exception as e:
            st.warning(f"⚠️ ERA5 API failed: {e}")
            return None
    
    def generate_indonesian_climate_data(self, lat, lon, start_date, end_date):
        """Generate realistic Indonesian climate data as fallback"""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Use coordinates for consistent seed
        seed = int(abs(lat * lon * 1000)) % 2**32
        np.random.seed(seed)
        
        st.info("🔄 Generating realistic Indonesian climate model data...")
        
        precip_values = []
        
        for date in dates:
            month = date.month
            
            # Indonesian monsoon patterns
            if month in [11, 12, 1, 2, 3, 4]:  # Wet season
                wet_season_factor = 3.0
                dry_day_prob = 0.15
            else:  # Dry season (May-Oct)
                wet_season_factor = 0.7
                dry_day_prob = 0.5
            
            # Equatorial factor (Indonesia is near equator)
            equatorial_factor = 1.4
            
            # Java/Sumatra vs Eastern Indonesia
            if lon < 115:  # Western Indonesia (Java, Sumatra)
                regional_factor = 1.2
            else:  # Eastern Indonesia
                regional_factor = 0.9
            
            if np.random.random() < dry_day_prob:
                daily_precip = 0
            else:
                base_precip = np.random.gamma(1.8, 6) * wet_season_factor * equatorial_factor * regional_factor
                
                # Occasional heavy rainfall events
                if np.random.random() < 0.08:  # 8% chance
                    base_precip *= np.random.uniform(3, 8)
                
                daily_precip = max(0, base_precip)
            
            precip_values.append(daily_precip)
        
        df = pd.DataFrame({
            'precipitation': precip_values
        }, index=dates)
        
        return {
            'source': 'Indonesian Climate Model (Synthetic)',
            'data': df,
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'note': 'Using realistic synthetic data based on Indonesian monsoon patterns',
            'climate_features': ['Monsoon seasons', 'Equatorial climate', 'Regional variations']
        }
    
    def prepare_spi_dataset(self, lat, lon, years_back=30):
        """Prepare complete dataset for SPI calculation using working APIs"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        st.header(f"🌧️ Collecting Precipitation Data")
        st.write(f"**Location:** {lat:.4f}°, {lon:.4f}°")
        st.write(f"**Period:** {years_back} years ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        # Try Open-Meteo Archive API first (tested working)
        dataset = self.get_open_meteo_data(lat, lon, start_date, end_date)
        
        if dataset is None:
            # Try ERA5 as backup
            dataset = self.get_era5_data(lat, lon, start_date, end_date)
        
        if dataset is None:
            # Use synthetic Indonesian climate model
            st.warning("⚠️ APIs unavailable, using synthetic Indonesian climate model")
            dataset = self.generate_indonesian_climate_data(lat, lon, start_date, end_date)
        
        if dataset and 'data' in dataset:
            # Convert daily to monthly data for SPI calculation
            daily_data = dataset['data']
            monthly_data = daily_data.resample('M').sum()
            
            dataset['data'] = monthly_data
            dataset['original_daily_points'] = len(daily_data)
            dataset['monthly_points'] = len(monthly_data)
            
            # Data quality info
            st.success(f"✅ Data prepared successfully!")
            st.write(f"📊 **Daily data points:** {len(daily_data)}")
            st.write(f"📅 **Monthly data points:** {len(monthly_data)}")
            st.write(f"💧 **Total precipitation:** {daily_data['precipitation'].sum():.1f} mm")
            st.write(f"📈 **Average monthly:** {monthly_data['precipitation'].mean():.1f} mm")
            
            return dataset
        
        return None

# ============================================================================
# SPI CALCULATOR CLASS
# ============================================================================

class SPICalculator:
    """SPI Calculator using proper statistical methods"""
    
    def __init__(self, timescales=[1, 3, 6, 12]):
        self.timescales = timescales
        self.fitted_params = {}
    
    def fit_gamma_distribution(self, data):
        """Fit gamma distribution using maximum likelihood estimation with improved error handling"""
        # Remove zero and negative values for gamma fitting
        non_zero_data = data[data > 0]
        
        if len(non_zero_data) < 10:
            st.warning(f"Only {len(non_zero_data)} non-zero values available for gamma fitting")
            # Use simple approximation for insufficient data
            mean_val = np.mean(data) if len(data) > 0 else 1.0
            return {
                'alpha': 1.0,
                'beta': max(mean_val, 1.0),
                'prob_zero': np.sum(data <= 0) / len(data) if len(data) > 0 else 0.0,
                'n_zeros': np.sum(data <= 0),
                'n_total': len(data)
            }
        
        # Method of moments for initial estimates
        mean_val = np.mean(non_zero_data)
        var_val = np.var(non_zero_data)
        
        if var_val <= 0 or mean_val <= 0:
            st.warning("Invalid variance or mean for gamma fitting")
            return {
                'alpha': 1.0,
                'beta': 1.0,
                'prob_zero': np.sum(data <= 0) / len(data),
                'n_zeros': np.sum(data <= 0),
                'n_total': len(data)
            }
        
        # Initial parameter estimates using method of moments
        beta_init = var_val / mean_val  # scale parameter
        alpha_init = mean_val / beta_init  # shape parameter
        
        # Ensure reasonable initial values
        alpha_init = max(0.1, min(alpha_init, 100))
        beta_init = max(0.1, min(beta_init, 1000))
        
        # MLE optimization with better error handling
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0.01 or beta <= 0.01 or alpha > 100 or beta > 1000:
                return 1e10  # Large penalty for invalid parameters
            try:
                ll = np.sum(stats.gamma.logpdf(non_zero_data, a=alpha, scale=beta))
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        try:
            result = minimize(neg_log_likelihood, [alpha_init, beta_init], 
                            method='L-BFGS-B', 
                            bounds=[(0.01, 100), (0.01, 1000)])
            
            if result.success:
                alpha_opt, beta_opt = result.x
            else:
                st.warning("Optimization failed, using method of moments")
                alpha_opt, beta_opt = alpha_init, beta_init
                
        except Exception as e:
            st.warning(f"MLE optimization error: {e}, using method of moments")
            alpha_opt, beta_opt = alpha_init, beta_init
        
        # Calculate probability of zero precipitation
        prob_zero = np.sum(data <= 0) / len(data)
        
        return {
            'alpha': float(alpha_opt),
            'beta': float(beta_opt),
            'prob_zero': float(prob_zero),
            'n_zeros': int(np.sum(data <= 0)),
            'n_total': int(len(data))
        }
    
    def calculate_spi_single_timescale(self, precipitation_data, timescale_months):
        """Calculate SPI for a single timescale with improved error handling"""
        # Create rolling sums for the specified timescale
        rolling_precip = precipitation_data.rolling(
            window=timescale_months, min_periods=timescale_months
        ).sum().dropna()
        
        if len(rolling_precip) < 30:  # Minimum 30 observations recommended
            st.warning(f"Insufficient data for {timescale_months}-month SPI calculation")
            return pd.Series(index=rolling_precip.index, dtype=float)
        
        try:
            # Fit gamma distribution
            gamma_params = self.fit_gamma_distribution(rolling_precip.values)
            self.fitted_params[f'{timescale_months}month'] = gamma_params
            
            # Calculate SPI values
            spi_values = []
            
            for i, precip_val in enumerate(rolling_precip.values):
                if precip_val <= 0:
                    # Handle zero or negative precipitation
                    cum_prob = gamma_params['prob_zero'] / 2  # Small probability for zero
                else:
                    # Cumulative probability from gamma distribution
                    try:
                        gamma_cdf = stats.gamma.cdf(precip_val, 
                                                  a=gamma_params['alpha'], 
                                                  scale=gamma_params['beta'])
                        # Adjust for probability of zero
                        cum_prob = gamma_params['prob_zero'] + (1 - gamma_params['prob_zero']) * gamma_cdf
                    except:
                        cum_prob = 0.5  # Default to normal conditions if calculation fails
                
                # Ensure probability is within valid range
                cum_prob = np.clip(cum_prob, 0.001, 0.999)
                
                # Transform to standard normal (SPI)
                try:
                    spi_val = stats.norm.ppf(cum_prob)
                    # Check for invalid SPI values
                    if np.isnan(spi_val) or np.isinf(spi_val):
                        spi_val = 0.0  # Default to normal if calculation fails
                except:
                    spi_val = 0.0
                
                spi_values.append(float(spi_val))
            
            spi_series = pd.Series(spi_values, index=rolling_precip.index, name=f'SPI_{timescale_months}')
            
            return spi_series
            
        except Exception as e:
            st.error(f"Error in SPI calculation for {timescale_months}m: {e}")
            # Return series with zeros if calculation fails
            return pd.Series([0.0] * len(rolling_precip), index=rolling_precip.index, name=f'SPI_{timescale_months}')
    
    def calculate_spi_all_timescales(self, precipitation_data):
        """Calculate SPI for all specified timescales"""
        results = {}
        
        if not isinstance(precipitation_data.index, pd.DatetimeIndex):
            raise ValueError("Precipitation data must have datetime index")
        
        for timescale in self.timescales:
            try:
                spi_series = self.calculate_spi_single_timescale(precipitation_data, timescale)
                results[f'SPI_{timescale}'] = spi_series
            except Exception as e:
                st.warning(f"Failed to calculate SPI for {timescale}-month timescale: {e}")
                results[f'SPI_{timescale}'] = pd.Series(index=precipitation_data.index, dtype=float)
        
        return pd.DataFrame(results)
    
    def classify_drought_intensity(self, spi_values):
        """Classify drought intensity based on SPI values"""
        def classify_single_value(spi):
            spi_val = safe_scalar_value(spi)
            
            if pd.isna(spi_val):
                return 'No Data'
            elif spi_val >= 2.0:
                return 'Extremely Wet'
            elif spi_val >= 1.5:
                return 'Very Wet'
            elif spi_val >= 1.0:
                return 'Moderately Wet'
            elif spi_val >= -1.0:
                return 'Near Normal'
            elif spi_val >= -1.5:
                return 'Moderately Dry'
            elif spi_val >= -2.0:
                return 'Severely Dry'
            else:
                return 'Extremely Dry'
        
        if isinstance(spi_values, pd.Series):
            return spi_values.apply(classify_single_value)
        else:
            return classify_single_value(spi_values)
    
    def get_drought_statistics(self, spi_data):
        """Calculate drought event statistics"""
        stats_dict = {}
        
        for column in spi_data.columns:
            spi_series = spi_data[column].dropna()
            if len(spi_series) == 0:
                continue
                
            drought_mask = spi_series < -1.0
            
            drought_events = []
            in_drought = False
            start_date = None
            
            for date, is_drought in drought_mask.items():
                if is_drought and not in_drought:
                    in_drought = True
                    start_date = date
                elif not is_drought and in_drought:
                    in_drought = False
                    event_data = spi_series[start_date:date]
                    drought_events.append({
                        'start': start_date,
                        'end': date,
                        'duration': len(event_data),
                        'severity': float(abs(event_data.sum())),  # Ensure scalar
                        'intensity': float(event_data.min())      # Ensure scalar
                    })
            
            if in_drought:
                event_data = spi_series[start_date:]
                drought_events.append({
                    'start': start_date,
                    'end': spi_series.index[-1],
                    'duration': len(event_data),
                    'severity': float(abs(event_data.sum())),  # Ensure scalar
                    'intensity': float(event_data.min()),     # Ensure scalar
                    'ongoing': True
                })
            
            # Calculate statistics with safe scalar conversion
            if drought_events:
                avg_duration = np.mean([e['duration'] for e in drought_events])
                max_severity = max([e['severity'] for e in drought_events])
                min_intensity = min([e['intensity'] for e in drought_events])
            else:
                avg_duration = 0
                max_severity = 0
                min_intensity = 0
            
            stats_dict[column] = {
                'total_events': len(drought_events),
                'avg_duration': float(avg_duration),      # Ensure scalar
                'max_severity': float(max_severity),      # Ensure scalar
                'min_intensity': float(min_intensity),    # Ensure scalar
                'events': drought_events
            }
        
        return stats_dict

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def display_results(dataset, spi_results, drought_stats, lat, lon, spi_calculator_instance):
    """Display analysis results with enhanced visualizations"""
    
    st.header("📊 Current Drought Conditions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_values = spi_results.iloc[-1] if len(spi_results) > 0 else {}
    
    for i, (col, timescale) in enumerate(zip([col1, col2, col3, col4], spi_results.columns[:4])):
        if timescale in latest_values:
            spi_val = safe_scalar_value(latest_values[timescale])
            drought_class = spi_calculator_instance.classify_drought_intensity(spi_val)
            
            with col:
                st.metric(
                    label=f"{timescale.replace('SPI_', '')} Month SPI",
                    value=f"{spi_val:.2f}" if not pd.isna(spi_val) else "N/A",
                    delta=drought_class
                )
    
    # Time series visualization
    create_spi_time_series_chart(spi_results, spi_calculator_instance)
    
    # Drought statistics
    create_drought_statistics_table(drought_stats)
    
    # Geographic visualization
    create_geographic_map(lat, lon, latest_values, dataset, spi_calculator_instance)
    
    # Data information
    display_data_information(dataset, spi_calculator_instance)

def create_spi_time_series_chart(spi_results, spi_calculator_instance):
    """Create the SPI time series chart with proper error handling"""
    st.header("📈 SPI Time Series Analysis")
    
    # Debug: Check if we have valid data for plotting
    if spi_results.empty:
        st.error("❌ No SPI data available for plotting")
        return
    
    # Check for valid, non-constant data
    valid_columns = []
    for col in spi_results.columns:
        series = spi_results[col].dropna()
        if len(series) > 1 and series.nunique() > 1:
            valid_columns.append(col)
        else:
            st.warning(f"⚠️ {col} has insufficient or constant data for plotting")
    
    if not valid_columns:
        st.error("❌ No valid SPI data for time series plotting")
        
        # Show alternative simple chart
        st.subheader("🔧 Data Diagnostic Chart")
        fig_diag = go.Figure()
        
        for i, col in enumerate(spi_results.columns):
            fig_diag.add_trace(go.Scatter(
                x=list(range(len(spi_results[col]))),
                y=spi_results[col].fillna(0),
                mode='lines+markers',
                name=col,
                line=dict(width=3)
            ))
        
        fig_diag.update_layout(
            title="SPI Data Diagnostic (Index vs Value)",
            xaxis_title="Data Point Index",
            yaxis_title="SPI Value",
            height=400
        )
        
        st.plotly_chart(fig_diag, use_container_width=True)
        return
    
    # Create time series plot with valid data only
    fig = make_subplots(
        rows=len(valid_columns), cols=1,
        subplot_titles=[f"{col.replace('SPI_', '')} Month SPI" for col in valid_columns],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, column in enumerate(valid_columns):
        # Get non-null data
        series_data = spi_results[column].dropna()
        
        if len(series_data) == 0:
            continue
            
        # Add SPI line
        fig.add_trace(
            go.Scatter(
                x=series_data.index,
                y=series_data.values,
                mode='lines',
                name=f"{column.replace('SPI_', '')} Month",
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=True
            ),
            row=i+1, col=1
        )
        
        # Add drought threshold lines
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     line_width=1, opacity=0.5, row=i+1, col=1)
        fig.add_hline(y=-1.0, line_dash="dash", line_color="orange", 
                     line_width=1, opacity=0.7, row=i+1, col=1)
        fig.add_hline(y=-1.5, line_dash="dash", line_color="red",
                     line_width=1, opacity=0.7, row=i+1, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="darkred",
                     line_width=1, opacity=0.7, row=i+1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="lightblue",
                     line_width=1, opacity=0.7, row=i+1, col=1)
        fig.add_hline(y=2.0, line_dash="dash", line_color="blue",
                     line_width=1, opacity=0.7, row=i+1, col=1)
        
        # Add colored background regions
        fig.add_hrect(y0=-4, y1=-2, fillcolor="darkred", opacity=0.1, 
                     line_width=0, row=i+1, col=1)
        fig.add_hrect(y0=-2, y1=-1.5, fillcolor="red", opacity=0.1, 
                     line_width=0, row=i+1, col=1)
        fig.add_hrect(y0=-1.5, y1=-1, fillcolor="orange", opacity=0.1, 
                     line_width=0, row=i+1, col=1)
        fig.add_hrect(y0=-1, y1=1, fillcolor="lightgreen", opacity=0.1, 
                     line_width=0, row=i+1, col=1)
        fig.add_hrect(y0=1, y1=2, fillcolor="lightblue", opacity=0.1, 
                     line_width=0, row=i+1, col=1)
        fig.add_hrect(y0=2, y1=4, fillcolor="blue", opacity=0.1, 
                     line_width=0, row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        height=max(300 * len(valid_columns), 400),
        title_text="Standard Precipitation Index (SPI) Time Series",
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Date", 
        row=len(valid_columns), 
        col=1,
        tickangle=45
    )
    
    for i in range(len(valid_columns)):
        fig.update_yaxes(
            title_text="SPI Value",
            range=[-3.5, 3.5],
            row=i+1,
            col=1
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics below the chart
    st.subheader("📊 Current SPI Values Summary")
    
    if len(valid_columns) > 0:
        summary_data = []
        latest_values = spi_results.iloc[-1] if len(spi_results) > 0 else {}
        
        for col in valid_columns:
            if col in latest_values:
                val = safe_scalar_value(latest_values[col])
                classification = spi_calculator_instance.classify_drought_intensity(val)
                summary_data.append({
                    'Timescale': col.replace('SPI_', '') + ' months',
                    'Current SPI': f"{val:.2f}" if not pd.isna(val) else "N/A",
                    'Classification': classification,
                    'Data Points': spi_results[col].count()
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

def create_drought_statistics_table(drought_stats):
    """Create drought statistics table"""
    st.header("📋 Drought Event Statistics")
    
    # Create statistics table
    stats_data = []
    for timescale, stats in drought_stats.items():
        stats_data.append({
            'Timescale': timescale.replace('SPI_', '') + ' Month',
            'Total Events': stats['total_events'],
            'Avg Duration (months)': f"{safe_scalar_value(stats['avg_duration']):.1f}",
            'Max Severity': f"{safe_scalar_value(stats['max_severity']):.2f}",
            'Min Intensity': f"{safe_scalar_value(stats['min_intensity']):.2f}"
        })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

def create_geographic_map(lat, lon, latest_values, dataset, spi_calculator_instance):
    """Create geographic map visualization"""
    st.header("🗺️ Geographic Context")
    
    # Create map
    m = folium.Map(location=[lat, lon], zoom_start=8)
    
    latest_spi_3month = safe_scalar_value(latest_values.get('SPI_3', 0)) if 'SPI_3' in latest_values else 0
    drought_class = spi_calculator_instance.classify_drought_intensity(latest_spi_3month)
    
    # Color based on drought intensity
    color_map = {
        'Extremely Dry': 'darkred',
        'Severely Dry': 'red', 
        'Moderately Dry': 'orange',
        'Near Normal': 'green',
        'Moderately Wet': 'lightblue',
        'Very Wet': 'blue',
        'Extremely Wet': 'darkblue'
    }
    
    marker_color = color_map.get(drought_class, 'gray')
    
    folium.Marker(
        [lat, lon],
        popup=f"""
        <b>Drought Monitoring Point</b><br>
        Coordinates: {lat:.4f}, {lon:.4f}<br>
        3-Month SPI: {latest_spi_3month:.2f}<br>
        Condition: {drought_class}<br>
        Data Source: {dataset['source']}
        """,
        icon=folium.Icon(color=marker_color, icon='tint')
    ).add_to(m)
    
    folium.Circle(
        [lat, lon],
        radius=50000,
        popup=f"Regional Analysis Area",
        color=marker_color,
        fill=True,
        opacity=0.3
    ).add_to(m)
    
    folium_static(m)

def display_data_information(dataset, spi_calculator_instance):
    """Display data source information"""
    st.header("ℹ️ Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Data Source:** {dataset['source']}
        
        **Location:** {dataset.get('lat', 'N/A')}°N, {dataset.get('lon', 'N/A')}°E
        
        **Analysis Period:** {dataset['period']}
        
        **Monthly Data Points:** {len(dataset['data'])}
        """)
    
    with col2:
        st.info(f"""
        **SPI Calculation Method:** Gamma Distribution Fitting
        
        **Timescales Analyzed:** {', '.join([str(t) for t in spi_calculator_instance.timescales])} months
        
        **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        
        **Quality Score:** {'Good' if len(dataset['data']) > 360 else 'Fair'}
        """)

def test_api_endpoints(lat, lon):
    """Test API endpoints for status checking"""
    results = {}
    
    # Test Open-Meteo Archive
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
            'daily': 'precipitation_sum'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results['Open-Meteo Archive'] = {'status': 'SUCCESS'}
        else:
            results['Open-Meteo Archive'] = {'status': 'FAILED', 'error': f"HTTP {response.status_code}"}
    except Exception as e:
        results['Open-Meteo Archive'] = {'status': 'ERROR', 'error': str(e)}
    
    # Test ERA5
    try:
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
            'daily': 'precipitation_sum'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results['ERA5 Reanalysis'] = {'status': 'SUCCESS'}
        else:
            results['ERA5 Reanalysis'] = {'status': 'FAILED', 'error': f"HTTP {response.status_code}"}
    except Exception as e:
        results['ERA5 Reanalysis'] = {'status': 'ERROR', 'error': str(e)}
    
    # Synthetic model always works
    results['Indonesian Climate Model'] = {'status': 'SUCCESS'}
    
    return results

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Drought Monitoring System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .drought-severe { color: #d62728; }
    .drought-moderate { color: #ff7f0e; }
    .drought-mild { color: #ffbb78; }
    .normal { color: #2ca02c; }
    .wet { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🌧️ Drought Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time drought monitoring using Standard Precipitation Index (SPI) with working APIs**")
    
    # Initialize components
    data_collector = IntegratedClimateDataCollector()
    spi_calculator = SPICalculator(timescales=[1, 3, 6, 12])
    
    # Sidebar controls
    with st.sidebar:
        st.header("📍 Location Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", -90.0, 90.0, -6.1750, format="%.4f")
        with col2:
            longitude = st.number_input("Longitude", -180.0, 180.0, 106.8283, format="%.4f")
        
        preset_locations = {
            "Jakarta, Indonesia": (-6.1750, 106.8283),
            "Surabaya, Indonesia": (-7.2504, 112.7688),
            "Medan, Indonesia": (3.5952, 98.6722),
            "Bandung, Indonesia": (-6.9175, 107.6191),
            "Makassar, Indonesia": (-5.1400, 119.4221),
            "Yogyakarta, Indonesia": (-7.7956, 110.3695),
            "Denpasar, Bali": (-8.6500, 115.2167),
            "Palembang, Indonesia": (-2.9761, 104.7754),
            "New York, USA": (40.7128, -74.0060),
            "London, UK": (51.5074, -0.1278)
        }
        
        selected_location = st.selectbox("Or select preset location:", 
                                       ["Custom"] + list(preset_locations.keys()))
        
        if selected_location != "Custom":
            latitude, longitude = preset_locations[selected_location]
            st.info(f"Selected: {selected_location}")
        
        st.header("⚙️ Analysis Settings")
        
        years_back = st.slider("Years of historical data", 5, 30, 15)
        
        selected_timescales = st.multiselect(
            "SPI Timescales (months)",
            [1, 3, 6, 9, 12, 24],
            default=[3, 6, 12]
        )
        
        # Analysis buttons
        analyze_button = st.button("🔍 Analyze Drought Conditions", type="primary")
        test_button = st.button("🧪 Test with Sample Data", help="Use synthetic data to test SPI calculation")
        
        # API Status Section
        st.header("🌐 API Status")
        
        if st.button("Test API Endpoints"):
            with st.spinner("Testing APIs..."):
                test_results = test_api_endpoints(latitude, longitude)
                
                for api_name, result in test_results.items():
                    if result['status'] == 'SUCCESS':
                        st.success(f"✅ {api_name}: Working")
                    else:
                        st.error(f"❌ {api_name}: {result.get('error', 'Failed')}")
        
        st.info("""
        **Available APIs:**
        - Open-Meteo Archive ✅
        - ERA5 Reanalysis ✅
        - Indonesian Climate Model ✅
        """)
    
    # Main content area
    if analyze_button or test_button:
        try:
            with st.spinner("🔄 Analyzing drought conditions..."):
                
                if test_button:
                    # Create test data for debugging
                    st.info("🧪 Using synthetic test data for debugging")
                    dataset = create_test_dataset(latitude, longitude, years_back)
                else:
                    # Collect data using working APIs
                    dataset = data_collector.prepare_spi_dataset(latitude, longitude, years_back)
                
                if dataset and 'data' in dataset:
                    st.header("🧮 Calculating SPI Indices")
                    
                    # Show data summary before SPI calculation
                    with st.expander("📊 Data Summary", expanded=False):
                        monthly_data = dataset['data']
                        st.write(f"**Monthly data shape:** {monthly_data.shape}")
                        st.write(f"**Date range:** {monthly_data.index.min()} to {monthly_data.index.max()}")
                        st.write(f"**Precipitation stats:**")
                        st.write(f"- Mean: {monthly_data['precipitation'].mean():.2f} mm")
                        st.write(f"- Std: {monthly_data['precipitation'].std():.2f} mm")
                        st.write(f"- Min: {monthly_data['precipitation'].min():.2f} mm")
                        st.write(f"- Max: {monthly_data['precipitation'].max():.2f} mm")
                        st.write(f"- Zero months: {(monthly_data['precipitation'] == 0).sum()}")
                        
                        # Show sample data
                        st.write("**Sample data (last 12 months):**")
                        st.dataframe(monthly_data.tail(12))
                    
                    # Calculate SPI
                    spi_calculator.timescales = selected_timescales
                    
                    # Add progress tracking for SPI calculation
                    spi_progress = st.progress(0)
                    spi_status = st.empty()
                    
                    spi_results = pd.DataFrame()
                    
                    for i, timescale in enumerate(selected_timescales):
                        spi_status.text(f"Calculating {timescale}-month SPI...")
                        try:
                            spi_series = spi_calculator.calculate_spi_single_timescale(dataset['data'], timescale)
                            spi_results[f'SPI_{timescale}'] = spi_series
                            spi_progress.progress((i + 1) / len(selected_timescales))
                        except Exception as e:
                            st.error(f"Failed to calculate {timescale}-month SPI: {e}")
                    
                    spi_status.empty()
                    spi_progress.empty()
                    
                    if not spi_results.empty:
                        st.success(f"✅ SPI calculated for {len(selected_timescales)} timescales")
                        
                        # Show SPI summary
                        with st.expander("📈 SPI Summary", expanded=False):
                            st.write("**SPI Data Shape:**", spi_results.shape)
                            st.write("**SPI Statistics:**")
                            st.dataframe(spi_results.describe())
                            
                            # Check for constant values
                            for col in spi_results.columns:
                                unique_vals = spi_results[col].nunique()
                                st.write(f"- {col}: {unique_vals} unique values")
                                if unique_vals <= 5:
                                    st.warning(f"⚠️ {col} has very few unique values - check data quality")
                        
                        # Get drought statistics
                        drought_stats = spi_calculator.get_drought_statistics(spi_results)
                        
                        # Display results
                        display_results(dataset, spi_results, drought_stats, latitude, longitude, spi_calculator)
                    else:
                        st.error("❌ Failed to calculate any SPI values")
                else:
                    st.error("❌ Failed to collect precipitation data")
                    
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            
            with st.expander("🔍 Debug Information"):
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# ADDITIONAL INFO SECTIONS
# ============================================================================

def show_app_info():
    """Show application information"""
    st.markdown("---")
    st.header("ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 **Purpose**
        This application provides real-time drought monitoring using the Standard Precipitation Index (SPI), 
        a meteorological drought indicator recommended by the World Meteorological Organization.
        
        ### 📊 **Features**
        - Real-time precipitation data collection
        - SPI calculation for multiple timescales
        - Interactive drought visualizations
        - Historical drought event analysis
        - Geographic drought mapping
        """)
    
    with col2:
        st.markdown("""
        ### 🌐 **Data Sources**
        - **Open-Meteo Archive API**: Historical weather data (1940-present)
        - **ERA5 Reanalysis**: High-quality reanalysis data
        - **Indonesian Climate Model**: Synthetic data based on monsoon patterns
        
        ### 📚 **Scientific Basis**
        - Gamma distribution fitting for precipitation data
        - Standard normal transformation for SPI calculation
        - WMO-recommended drought classification system
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
    
    # Show app information

    show_app_info()
