"""
Strong App Workout Analysis Dashboard.

This module provides analysis and visualization of workout data from the Strong App,
focusing on workout frequency, exercise progression, and one-rep max calculations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Filter warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Type aliases
DateSeries = pd.Series
DataFrame = pd.DataFrame

@dataclass(frozen=True)
class WorkoutData:
    """Immutable container for workout data and analysis results."""
    raw_data: DataFrame
    clean_data: DataFrame
    date_spine: DataFrame
    available_years: List[int]
    
    @property
    def exercises(self) -> List[str]:
        """Get unique exercise names."""
        return sorted(self.raw_data['Exercise Name'].unique().tolist())
    
    @property
    def workout_names(self) -> List[str]:
        """Get unique workout names."""
        return sorted(self.raw_data['Workout Name'].unique().tolist())

def create_date_spine(start_date: datetime, end_date: datetime) -> DataFrame:
    """Create a complete date range with metadata.
    
    Args:
        start_date: Start date for the spine
        end_date: End date for the spine
        
    Returns:
        DataFrame with date range and associated metadata
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'date_str': dates.strftime('%Y-%m-%d'),
        'day_name': dates.strftime('%A'),
        'day_name_short': dates.strftime('%a'),
        'day_of_week': dates.dayofweek,
        'week_of_year': dates.isocalendar().week,
        'month': dates.month,
        'month_name': dates.strftime('%B'),
        'year': dates.year,
        'year_month': dates.strftime('%Y-%m'),
        'year_week': dates.strftime('%Y-W%V'),
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
    })

def calculate_one_rep_max(weight: float, reps: int) -> float:
    """Calculate estimated one rep max using Brzycki formula.
    
    Args:
        weight: Weight used in the set
        reps: Number of repetitions performed
        
    Returns:
        Estimated one rep max
    """
    return weight * (36 / (37 - reps))

def load_workout_data(file_path: str) -> WorkoutData:
    """Load and prepare workout data from CSV file.
    
    Args:
        file_path: Path to the Strong App CSV export
        
    Returns:
        WorkoutData container with processed data
    """
    # Read raw data
    raw_df = pd.read_csv(file_path)
    
    # Convert date column
    raw_df['date'] = pd.to_datetime(raw_df['Date']).dt.normalize()
    
    # Calculate 1RM for valid sets
    mask = raw_df[['Weight', 'Reps']].notna().all(axis=1)
    raw_df.loc[mask, 'one_rep_max'] = raw_df[mask].apply(
        lambda x: calculate_one_rep_max(x['Weight'], x['Reps']), 
        axis=1
    )
    
    # Get available years
    available_years = sorted(raw_df['date'].dt.year.unique().tolist())
    
    # Create date spine for full date range
    start_date = raw_df['date'].min()
    end_date = raw_df['date'].max()
    date_spine = create_date_spine(start_date, end_date)
    
    return WorkoutData(
        raw_data=raw_df,
        clean_data=raw_df,  # Will be filtered by year when needed
        date_spine=date_spine,
        available_years=available_years
    )

def filter_data_by_year(data: WorkoutData, year: int) -> Tuple[DataFrame, DataFrame]:
    """Filter workout data for a specific year.
    
    Args:
        data: WorkoutData container
        year: Year to filter for
        
    Returns:
        Tuple of (date_spine_with_counts, filtered_workout_data)
    """
    # Create year's date spine
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31")
    date_spine = create_date_spine(start_date, end_date)
    
    # Filter workout data
    year_data = data.clean_data[data.clean_data['date'].dt.year == year].copy()
    
    if len(year_data) == 0:
        date_spine['workout_count'] = 0
        return date_spine, pd.DataFrame()
    
    # Count unique workouts per day
    daily_workouts = (year_data
                     .groupby('date')['Workout Name']
                     .nunique()
                     .reset_index(name='workout_count'))
    
    # Merge with date spine
    date_spine_with_counts = pd.merge(
        date_spine,
        daily_workouts,
        on='date',
        how='left'
    ).fillna({'workout_count': 0})
    
    return date_spine_with_counts, year_data

def get_exercise_stats(workout_data: DataFrame, exercise_name: str) -> Tuple[float, float]:
    """Calculate current and best 1RM for an exercise.
    
    Args:
        workout_data: Filtered workout data
        exercise_name: Name of the exercise
        
    Returns:
        Tuple of (current_1rm, best_1rm)
    """
    exercise_data = workout_data[workout_data['Exercise Name'] == exercise_name]
    
    if len(exercise_data) == 0:
        return 0.0, 0.0
    
    best_1rm = exercise_data['one_rep_max'].max()
    
    # Get current (last 7 days) 1RM
    latest_date = exercise_data['date'].max()
    current_data = exercise_data[
        exercise_data['date'] >= latest_date - pd.Timedelta(days=7)
    ]
    current_1rm = current_data['one_rep_max'].max()
    
    return current_1rm, best_1rm

# Visualization functions
def create_monthly_frequency_chart(df: DataFrame) -> go.Figure:
    """Create monthly workout frequency visualization.
    
    Args:
        df: DataFrame with workout counts
        
    Returns:
        Plotly figure object
    """
    monthly_counts = (df.groupby('year_month')['workout_count']
                     .sum()
                     .reset_index()
                     .sort_values('year_month'))
    
    fig = go.Figure(
        go.Bar(
            x=monthly_counts['year_month'],
            y=monthly_counts['workout_count'],
            text=monthly_counts['workout_count'].astype(int),
            textposition='auto',
            marker_color='#8a3ffc',
            hovertemplate="Month: %{x}<br>Workouts: %{y}<extra></extra>"
        )
    )
    
    fig.update_layout(
        title=dict(
            text="Monthly Workout Frequency",
            font=dict(size=24, color='#8a3ffc'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickangle=45,
            type='category',
            tickfont=dict(size=12, color='#4589ff')
        ),
        yaxis=dict(
            title="Number of Workouts",
            rangemode='tozero',
            tickfont=dict(size=12, color='#4589ff'),
            gridcolor='rgba(138, 63, 252, 0.1)'
        ),
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_weekday_frequency_chart(df: DataFrame) -> go.Figure:
    """Create weekday workout frequency visualization.
    
    Args:
        df: DataFrame with workout counts
        
    Returns:
        Plotly figure object
    """
    weekday_counts = (df[df['workout_count'] > 0]
                     .groupby(['day_name', 'day_of_week'])['workout_count']
                     .sum()
                     .reset_index()
                     .sort_values('day_of_week'))
    
    colors = ['#8a3ffc', '#33b1ff', '#007d79', '#ff7eb6', '#fa4d56', '#6fdc8c', '#4589ff']
    
    fig = go.Figure(
        go.Bar(
            x=weekday_counts['day_name'],
            y=weekday_counts['workout_count'],
            text=weekday_counts['workout_count'].astype(int),
            textposition='auto',
            marker_color=colors,
            hovertemplate="Day: %{x}<br>Workouts: %{y}<extra></extra>"
        )
    )
    
    fig.update_layout(
        title=dict(
            text="Workout Frequency by Day",
            font=dict(size=24, color='#8a3ffc'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            categoryorder='array',
            categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            tickfont=dict(size=12, color='#4589ff')
        ),
        yaxis=dict(
            title="Number of Workouts",
            rangemode='tozero',
            tickfont=dict(size=12, color='#4589ff'),
            gridcolor='rgba(138, 63, 252, 0.1)'
        ),
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_activity_heatmap(df: DataFrame) -> go.Figure:
    """Create workout activity heatmap visualization.
    
    Args:
        df: DataFrame with workout counts
        
    Returns:
        Plotly figure object
    """
    custom_colorscale = [
        [0, '#fff1f1'],
        [0.5, '#8a3ffc'],
        [1, '#4589ff']
    ]
    
    fig = go.Figure(
        go.Heatmap(
            x=df['week_of_year'],
            y=df['day_of_week'],
            z=df['workout_count'],
            colorscale=custom_colorscale,
            showscale=False,
            hoverongaps=False,
            xgap=3,
            ygap=3,
            hovertemplate="%{customdata}<br>Workouts: %{z}<extra></extra>",
            customdata=df['date_str']
        )
    )
    
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    fig.update_layout(
        title=dict(
            text="Activity Timeline",
            font=dict(size=24, color='#8a3ffc'),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            title="",
            ticktext=day_names,
            tickvals=list(range(7)),
            showgrid=False,
            tickfont=dict(size=14, color='#4589ff'),
            autorange="reversed",
            showline=False,
            side='left',
            ticklen=0,
            fixedrange=True
        ),
        height=300,
        autosize=True,
        margin=dict(l=50, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False
    )
    
    return fig

def create_exercise_progress_chart(
    workout_data: DataFrame,
    exercise_names: List[str]
) -> go.Figure:
    """Create multi-exercise progress visualization.
    
    Args:
        workout_data: DataFrame with workout data
        exercise_names: List of exercises to plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create date range
    year = workout_data['date'].dt.year.iloc[0]
    date_range = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq='D'
    )
    
    colors = ['#8a3ffc', '#33b1ff', '#007d79', '#ff7eb6', '#fa4d56', '#6fdc8c', '#4589ff']
    all_values = []
    
    for idx, exercise in enumerate(exercise_names):
        exercise_data = workout_data[workout_data['Exercise Name'] == exercise].copy()
        
        if len(exercise_data) == 0:
            continue
            
        # Calculate weekly stats
        exercise_data['week_start'] = exercise_data['date'].dt.to_period('W').dt.start_time
        weekly_stats = (exercise_data.groupby('week_start')
                       .agg({
                           'one_rep_max': ['max', 'mean', 'count'],
                           'date': 'first'
                       })
                       .reset_index())
        weekly_stats.columns = ['week_start', 'max_1rm', 'mean_1rm', 'count', 'date']
        
        weekly_stats = weekly_stats[weekly_stats['count'] >= 1].sort_values('week_start')
        weekly_stats['smooth_1rm'] = (weekly_stats['max_1rm']
                                    .rolling(window=3, center=True, min_periods=1)
                                    .mean())
        
        all_values.extend(weekly_stats['max_1rm'].tolist())
        
        # Add raw data points
        fig.add_trace(
            go.Scatter(
                x=weekly_stats['week_start'],
                y=weekly_stats['max_1rm'],
                name=f"{exercise} (Raw)",
                mode='markers',
                marker=dict(
                    color=colors[idx % len(colors)],
                    size=8,
                    opacity=0.3
                ),
                showlegend=False,
                hovertemplate=(
                    "Week of: %{x|%Y-%m-%d}<br>"
                    "1RM: %{y:.1f} lbs<br>"
                    "Sets: %{customdata[0]}<br>"
                    "Avg: %{customdata[1]:.1f} lbs"
                    "<extra></extra>"
                ),
                customdata=weekly_stats[['count', 'mean_1rm']]
            )
        )
        
        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=weekly_stats['week_start'],
                y=weekly_stats['smooth_1rm'],
                name=exercise,
                mode='lines',
                line=dict(
                    color=colors[idx % len(colors)],
                    width=3
                ),
                hovertemplate=(
                    "Week of: %{x|%Y-%m-%d}<br>"
                    "Smoothed 1RM: %{y:.1f} lbs"
                    "<extra></extra>"
                )
            )
        )
    
    # Calculate y-axis range
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_padding = (y_max - y_min) * 0.1
        y_min = max(0, y_min - y_padding)
        y_max = y_max + y_padding
    else:
        y_min, y_max = 0, 100
    
    # Add month markers
    month_starts = pd.date_range(start=date_range.min(), end=date_range.max(), freq='MS')
    for month_start in month_starts:
        fig.add_vline(
            x=month_start,
            line_width=1,
            line_dash="dash",
            line_color="rgba(138, 63, 252, 0.1)"
        )
    
    fig.update_layout(
        title=dict(
            text="One Rep Max Progression",
            font=dict(size=24, color='#8a3ffc'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickformat="%b %Y",
            dtick="M1",
            tickangle=45,
            range=[date_range.min(), date_range.max()],
            showgrid=False,
            tickmode='array',
            ticktext=[d.strftime('%b') for d in month_starts],
            tickvals=month_starts,
            tickfont=dict(size=12, color='#4589ff')
        ),
        yaxis=dict(
            title="Estimated 1RM (lbs)",
            showgrid=True,
            gridcolor='rgba(138, 63, 252, 0.1)',
            range=[y_min, y_max],
            zeroline=True,
            zerolinecolor='rgba(138, 63, 252, 0.2)',
            zerolinewidth=1,
            tickfont=dict(size=12, color='#4589ff')
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            font=dict(color='white')
        ),
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    return fig

def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Strong App Analysis",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
            .block-container {
                padding: 1rem 2rem;
            }
            .element-container {
                width: 100%;
            }
            h1 {
                color: #8a3ffc;
                font-weight: 600;
                font-size: 2.5rem;
                margin-bottom: 2rem;
            }
            h2 {
                color: #4589ff;
                font-weight: 500;
                font-size: 1.8rem;
                margin-top: 2rem;
            }
            [data-testid="stMetricValue"] {
                color: #8a3ffc;
                font-weight: 600;
            }
            [data-testid="stMetricLabel"] {
                color: #4589ff;
            }
            .stSelectbox label, .stMultiSelect label {
                color: #4589ff;
                font-weight: 500;
            }
            [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
                background-color: #fff;
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            }
        </style>
    """, unsafe_allow_html=True)

def display_data_overview(workout_data: DataFrame, year: int) -> None:
    """Display overview statistics of workout data.
    
    Args:
        workout_data: DataFrame with workout data
        year: Selected year
    """
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Workouts",
            len(workout_data['Workout Name'].unique())
        )
    with col2:
        st.metric(
            "Unique Exercises",
            len(workout_data['Exercise Name'].unique())
        )
    with col3:
        st.metric(
            "Active Days",
            len(workout_data['date'].unique())
        )

def display_exercise_analysis(
    workout_data: DataFrame,
    exercise_name: str,
    current_1rm: float,
    best_1rm: float
) -> None:
    """Display exercise-specific analysis.
    
    Args:
        workout_data: DataFrame with workout data
        exercise_name: Selected exercise name
        current_1rm: Current one rep max
        best_1rm: Best one rep max
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Week 1RM", f"{current_1rm:.1f} lbs")
    with col2:
        st.metric("Best 1RM", f"{best_1rm:.1f} lbs")

def parse_duration(duration_str: str) -> float:
    """Parse duration string into minutes.
    
    Args:
        duration_str: Duration string in format like "1h 5min" or "45min"
        
    Returns:
        Total minutes as float
    """
    total_minutes = 0
    
    # Extract hours if present
    if 'h' in duration_str:
        hours = float(duration_str.split('h')[0])
        total_minutes += hours * 60
        
    # Extract minutes if present
    if 'min' in duration_str:
        if 'h' in duration_str:
            minutes = float(duration_str.split('h')[1].split('min')[0])
        else:
            minutes = float(duration_str.split('min')[0])
        total_minutes += minutes
    
    return total_minutes

def calculate_streaks(workout_dates: pd.Series) -> dict:
    """Calculate daily, weekly, and monthly streaks."""
    print("\n=== Debug Streak Calculation ===")
    print(f"Input dates shape: {workout_dates.shape}")
    
    # Handle empty case
    if len(workout_dates) == 0:
        print("No workout dates found")
        return {
            'daily': {'days': 0, 'start': None, 'end': None},
            'weekly': {'weeks': 0, 'start': None, 'end': None},
            'monthly': {'months': 0, 'start': None, 'end': None}
        }
    
    # Sort unique dates
    workout_dates = pd.Series(workout_dates.unique()).sort_values()
    print("\nDate range:")
    print(f"First date: {workout_dates.iloc[0]}")
    print(f"Last date: {workout_dates.iloc[-1]}")
    
    # Daily streaks
    date_diff = workout_dates.diff()
    daily_streak_breaks = date_diff[date_diff > pd.Timedelta(days=1)].index
    print(f"\nFound {len(daily_streak_breaks)} daily streak breaks")
    
    # Calculate daily streaks
    daily_streaks = []
    start_idx = 0
    
    if len(daily_streak_breaks) == 0:
        daily_streaks.append({
            'length': len(workout_dates),
            'start': workout_dates.iloc[0],
            'end': workout_dates.iloc[-1]
        })
    else:
        for break_idx in daily_streak_breaks:
            daily_streaks.append({
                'length': break_idx - start_idx,
                'start': workout_dates.iloc[start_idx],
                'end': workout_dates.iloc[break_idx - 1]
            })
            start_idx = break_idx
        
        # Add the last streak
        daily_streaks.append({
            'length': len(workout_dates) - start_idx,
            'start': workout_dates.iloc[start_idx],
            'end': workout_dates.iloc[-1]
        })
    
    print("\nDaily streaks found:")
    for streak in daily_streaks:
        print(f"Length: {streak['length']}, {streak['start']} to {streak['end']}")
    
    # Weekly streaks (consecutive weeks with workouts)
    workout_weeks = pd.Series(workout_dates.dt.strftime('%Y-%W').unique()).sort_values()
    print(f"\nFound {len(workout_weeks)} unique weeks")
    
    if len(workout_weeks) == 0:
        print("No weekly data found")
        return {
            'daily': {'days': 0, 'start': None, 'end': None},
            'weekly': {'weeks': 0, 'start': None, 'end': None},
            'monthly': {'months': 0, 'start': None, 'end': None}
        }
    
    week_numbers = pd.Series(pd.to_datetime(workout_weeks + '-1', format='%Y-%W-%w'))
    week_diff = week_numbers.diff()
    weekly_streak_breaks = week_diff[week_diff > pd.Timedelta(days=7)].index
    print(f"Found {len(weekly_streak_breaks)} weekly streak breaks")
    
    weekly_streaks = []
    start_idx = 0
    
    if len(weekly_streak_breaks) == 0:
        first_week_workouts = workout_dates[workout_dates.dt.strftime('%Y-%W') == workout_weeks.iloc[0]]
        last_week_workouts = workout_dates[workout_dates.dt.strftime('%Y-%W') == workout_weeks.iloc[-1]]
        weekly_streaks.append({
            'length': len(week_numbers),
            'start': first_week_workouts.iloc[0],
            'end': last_week_workouts.iloc[-1]
        })
    else:
        for break_idx in weekly_streak_breaks:
            start_week = workout_weeks.iloc[start_idx]
            end_week = workout_weeks.iloc[break_idx - 1]
            start_week_workouts = workout_dates[workout_dates.dt.strftime('%Y-%W') == start_week]
            end_week_workouts = workout_dates[workout_dates.dt.strftime('%Y-%W') == end_week]
            
            weekly_streaks.append({
                'length': break_idx - start_idx,
                'start': start_week_workouts.iloc[0],
                'end': end_week_workouts.iloc[-1]
            })
            start_idx = break_idx
        
        # Add the last streak
        start_week = workout_weeks.iloc[start_idx]
        end_week = workout_weeks.iloc[-1]
        start_week_workouts = workout_dates[workout_dates.dt.strftime('%Y-%W') == start_week]
        end_week_workouts = workout_dates[workout_dates.dt.strftime('%Y-%W') == end_week]
        
        weekly_streaks.append({
            'length': len(week_numbers) - start_idx,
            'start': start_week_workouts.iloc[0],
            'end': end_week_workouts.iloc[-1]
        })
    
    print("\nWeekly streaks found:")
    for streak in weekly_streaks:
        print(f"Length: {streak['length']}, {streak['start']} to {streak['end']}")
    
    # Monthly streaks (consecutive months with workouts)
    workout_months = pd.Series(workout_dates.dt.strftime('%Y-%m').unique()).sort_values()
    print(f"\nFound {len(workout_months)} unique months")
    
    if len(workout_months) == 0:
        print("No monthly data found")
        return {
            'daily': {'days': 0, 'start': None, 'end': None},
            'weekly': {'weeks': 0, 'start': None, 'end': None},
            'monthly': {'months': 0, 'start': None, 'end': None}
        }
    
    month_numbers = pd.Series(pd.to_datetime(workout_months + '-01'))
    month_diff = month_numbers.diff()
    monthly_streak_breaks = month_diff[month_diff > pd.Timedelta(days=31)].index
    print(f"Found {len(monthly_streak_breaks)} monthly streak breaks")
    
    monthly_streaks = []
    start_idx = 0
    
    if len(monthly_streak_breaks) == 0:
        first_month_workouts = workout_dates[workout_dates.dt.strftime('%Y-%m') == workout_months.iloc[0]]
        last_month_workouts = workout_dates[workout_dates.dt.strftime('%Y-%m') == workout_months.iloc[-1]]
        monthly_streaks.append({
            'length': len(month_numbers),
            'start': first_month_workouts.iloc[0],
            'end': last_month_workouts.iloc[-1]
        })
    else:
        for break_idx in monthly_streak_breaks:
            start_month = workout_months.iloc[start_idx]
            end_month = workout_months.iloc[break_idx - 1]
            start_month_workouts = workout_dates[workout_dates.dt.strftime('%Y-%m') == start_month]
            end_month_workouts = workout_dates[workout_dates.dt.strftime('%Y-%m') == end_month]
            
            monthly_streaks.append({
                'length': break_idx - start_idx,
                'start': start_month_workouts.iloc[0],
                'end': end_month_workouts.iloc[-1]
            })
            start_idx = break_idx
        
        # Add the last streak
        start_month = workout_months.iloc[start_idx]
        end_month = workout_months.iloc[-1]
        start_month_workouts = workout_dates[workout_dates.dt.strftime('%Y-%m') == start_month]
        end_month_workouts = workout_dates[workout_dates.dt.strftime('%Y-%m') == end_month]
        
        monthly_streaks.append({
            'length': len(month_numbers) - start_idx,
            'start': start_month_workouts.iloc[0],
            'end': end_month_workouts.iloc[-1]
        })
    
    print("\nMonthly streaks found:")
    for streak in monthly_streaks:
        print(f"Length: {streak['length']}, {streak['start']} to {streak['end']}")
    
    # Get longest streaks
    longest_daily = max(daily_streaks, key=lambda x: x['length'])
    longest_weekly = max(weekly_streaks, key=lambda x: x['length'])
    longest_monthly = max(monthly_streaks, key=lambda x: x['length'])
    
    print("\nLongest streaks:")
    print(f"Daily: {longest_daily['length']} days")
    print(f"Weekly: {longest_weekly['length']} weeks")
    print(f"Monthly: {longest_monthly['length']} months")
    
    return {
        'daily': {
            'days': longest_daily['length'],
            'start': longest_daily['start'],
            'end': longest_daily['end']
        },
        'weekly': {
            'weeks': longest_weekly['length'],
            'start': longest_weekly['start'],
            'end': longest_weekly['end']
        },
        'monthly': {
            'months': longest_monthly['length'],
            'start': longest_monthly['start'],
            'end': longest_monthly['end']
        }
    }

def calculate_fun_facts(workout_data: DataFrame) -> dict:
    """Calculate fun and interesting statistics about workouts."""
    print("\n=== Debug Fun Facts Calculation ===")
    print(f"Input data shape: {workout_data.shape}")
    
    # Handle empty data case
    if len(workout_data) == 0:
        print("No workout data found")
        return {
            'favorite_exercise': {'name': 'No exercises', 'count': 0},
            'least_favorite': {'name': 'No exercises', 'count': 0},
            'longest_workout': {'duration': 0, 'date': None},
            'strongest_day': {'date': None, 'volume': 0},
            'most_common_hour': 0,
            'favorite_workout': 'No workouts',
            'streaks': calculate_streaks(pd.Series(dtype='datetime64[ns]'))
        }
    
    # Most and least frequent exercises
    print("\nCalculating exercise frequencies...")
    exercise_counts = (workout_data.groupby('Exercise Name').size()
                      .reset_index(name='count')
                      .sort_values('count', ascending=False))
    print(f"Exercise counts shape: {exercise_counts.shape}")
    print("Top 5 exercises:")
    print(exercise_counts.head())
    
    if len(exercise_counts) > 0:
        favorite_exercise = {
            'name': exercise_counts.iloc[0]['Exercise Name'],
            'count': exercise_counts.iloc[0]['count']
        }
        least_favorite = {
            'name': exercise_counts.iloc[-1]['Exercise Name'],
            'count': exercise_counts.iloc[-1]['count']
        }
    else:
        print("No exercise counts found")
        favorite_exercise = {'name': 'No exercises', 'count': 0}
        least_favorite = {'name': 'No exercises', 'count': 0}
    
    # Group by date and workout name
    print("\nCalculating workout durations...")
    workout_durations = workout_data.groupby(['date', 'Workout Name'])['Duration'].first()
    print(f"Workout durations shape: {workout_durations.shape}")
    print("Sample durations:")
    print(workout_durations.head())
    
    # Convert durations to minutes and filter out unreasonable values (>24 hours)
    print("\nConverting durations to minutes...")
    duration_minutes = workout_durations.apply(parse_duration)
    print("Sample duration minutes:")
    print(duration_minutes.head())
    
    reasonable_durations = duration_minutes[duration_minutes <= 24 * 60]  # Max 24 hours
    print(f"\nReasonable durations count: {len(reasonable_durations)}")
    if len(reasonable_durations) > 0:
        print("Sample reasonable durations:")
        print(reasonable_durations.head())
    
    if len(reasonable_durations) > 0:
        longest_workout_minutes = reasonable_durations.max()
        longest_workout_idx = reasonable_durations.idxmax()
        longest_workout_date = longest_workout_idx[0]  # First element of the tuple index
        print(f"\nLongest workout: {longest_workout_minutes} minutes on {longest_workout_date}")
    else:
        print("\nNo reasonable workout durations found")
        longest_workout_minutes = 0
        longest_workout_date = None
    
    # Strongest day (highest total volume)
    print("\nCalculating volume stats...")
    workout_data['volume'] = workout_data['Weight'] * workout_data['Reps']
    volume_by_date = workout_data.groupby('date')['volume'].sum().reset_index()
    print(f"Volume by date shape: {volume_by_date.shape}")
    print("Sample volumes:")
    print(volume_by_date.head())
    
    if len(volume_by_date) > 0:
        strongest_idx = volume_by_date['volume'].idxmax()
        strongest_day = {
            'date': volume_by_date.loc[strongest_idx, 'date'],
            'volume': volume_by_date.loc[strongest_idx, 'volume']
        }
        print(f"\nStrongest day: {strongest_day['date']} with volume {strongest_day['volume']}")
    else:
        print("\nNo volume data found")
        strongest_day = {'date': None, 'volume': 0}
    
    # Most consistent time
    print("\nCalculating workout times...")
    workout_times = pd.to_datetime(workout_data['Date']).dt.hour
    print(f"Workout times shape: {workout_times.shape}")
    print("Sample times:")
    print(workout_times.head())
    
    if len(workout_times) > 0:
        most_common_hour = workout_times.mode().iloc[0]
        print(f"Most common hour: {most_common_hour}")
    else:
        print("No workout times found")
        most_common_hour = 0
    
    # Favorite workout type
    print("\nCalculating favorite workout...")
    if len(workout_data) > 0:
        favorite_workout = workout_data['Workout Name'].mode().iloc[0]
        print(f"Favorite workout: {favorite_workout}")
    else:
        print("No workout names found")
        favorite_workout = 'No workouts'
    
    # Calculate streaks
    print("\nCalculating streaks...")
    streaks = calculate_streaks(workout_data['date'])
    print("Streak results:")
    print(streaks)
    
    return {
        'favorite_exercise': favorite_exercise,
        'least_favorite': least_favorite,
        'longest_workout': {
            'duration': longest_workout_minutes,
            'date': longest_workout_date
        },
        'strongest_day': strongest_day,
        'most_common_hour': most_common_hour,
        'favorite_workout': favorite_workout,
        'streaks': streaks
    }

def format_duration(minutes: float) -> str:
    """Format minutes into a readable duration string.
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted string like "1h 30m" or "45m"
    """
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    
    if hours > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{remaining_minutes}m"

def display_fun_facts(workout_data: DataFrame) -> None:
    """Display fun facts about workout data."""
    facts = calculate_fun_facts(workout_data)
    
    st.header("Fun Facts 🎯")
    
    # Create three columns for the first row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style='background-color: rgba(138, 63, 252, 0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: #8a3ffc; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Favorite Exercise 💪</h3>
                <p style='color: #4589ff; margin: 0; font-size: 1rem;'>{facts['favorite_exercise']['name']}</p>
                <p style='color: #4589ff; margin: 0; font-size: 0.9rem;'>Done {facts['favorite_exercise']['count']} times</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Format streak dates
        date_range = ""
        if facts['streaks']['monthly']['start'] is not None:
            start_date = pd.to_datetime(facts['streaks']['monthly']['start']).strftime('%b %d')
            end_date = pd.to_datetime(facts['streaks']['monthly']['end']).strftime('%b %d')
            date_range = f"{start_date} - {end_date}"
        
        # Format streak descriptions
        daily_desc = "day" if facts['streaks']['daily']['days'] == 1 else "days"
        weekly_desc = "week" if facts['streaks']['weekly']['weeks'] == 1 else "weeks"
        monthly_desc = "month" if facts['streaks']['monthly']['months'] == 1 else "months"
        
        st.markdown(
            f"""
            <div style='background-color: rgba(51, 177, 255, 0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: #33b1ff; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Workout Streaks 🔥</h3>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; text-align: center;'>
                    <div style='background-color: rgba(51, 177, 255, 0.1); padding: 0.5rem; border-radius: 0.3rem;'>
                        <div style='color: #4589ff; font-size: 1.2rem; font-weight: 600;'>{facts['streaks']['daily']['days']}</div>
                        <div style='color: #4589ff; font-size: 0.8rem;'>{daily_desc}</div>
                    </div>
                    <div style='background-color: rgba(51, 177, 255, 0.1); padding: 0.5rem; border-radius: 0.3rem;'>
                        <div style='color: #4589ff; font-size: 1.2rem; font-weight: 600;'>{facts['streaks']['weekly']['weeks']}</div>
                        <div style='color: #4589ff; font-size: 0.8rem;'>{weekly_desc}</div>
                    </div>
                    <div style='background-color: rgba(51, 177, 255, 0.1); padding: 0.5rem; border-radius: 0.3rem;'>
                        <div style='color: #4589ff; font-size: 1.2rem; font-weight: 600;'>{facts['streaks']['monthly']['months']}</div>
                        <div style='color: #4589ff; font-size: 0.8rem;'>{monthly_desc}</div>
                    </div>
                </div>
                <p style='color: #4589ff; margin: 0.5rem 0 0 0; font-size: 0.8rem; font-style: italic; text-align: center;'>
                    {date_range}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        # Format workout date
        workout_date = ""
        if facts['longest_workout']['date'] is not None:
            workout_date = f"on {pd.to_datetime(facts['longest_workout']['date']).strftime('%b %d')}"
        
        st.markdown(
            f"""
            <div style='background-color: rgba(0, 125, 121, 0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: #007d79; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Longest Workout 🏋️</h3>
                <p style='color: #4589ff; margin: 0; font-size: 1rem;'>{format_duration(facts['longest_workout']['duration'])}</p>
                <p style='color: #4589ff; margin: 0; font-size: 0.9rem;'>
                    {workout_date}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Create three columns for the second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Format strongest day date
        strongest_date = ""
        if facts['strongest_day']['date'] is not None:
            strongest_date = pd.to_datetime(facts['strongest_day']['date']).strftime('%B %d')
        
        st.markdown(
            f"""
            <div style='background-color: rgba(255, 126, 182, 0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: #ff7eb6; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Strongest Day 🏆</h3>
                <p style='color: #4589ff; margin: 0; font-size: 1rem;'>{strongest_date}</p>
                <p style='color: #4589ff; margin: 0; font-size: 0.9rem;'>
                    {facts['strongest_day']['volume']:,.0f} lbs total volume
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col5:
        am_pm = "AM" if facts['most_common_hour'] < 12 else "PM"
        hour_12 = facts['most_common_hour'] % 12
        hour_12 = 12 if hour_12 == 0 else hour_12
        st.markdown(
            f"""
            <div style='background-color: rgba(250, 77, 86, 0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: #fa4d56; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Favorite Time ⏰</h3>
                <p style='color: #4589ff; margin: 0; font-size: 1rem;'>{hour_12}:00 {am_pm}</p>
                <p style='color: #4589ff; margin: 0; font-size: 0.9rem;'>Most common workout time</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col6:
        st.markdown(
            f"""
            <div style='background-color: rgba(111, 220, 140, 0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: #6fdc8c; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Least Done Exercise 🤔</h3>
                <p style='color: #4589ff; margin: 0; font-size: 1rem;'>{facts['least_favorite']['name']}</p>
                <p style='color: #4589ff; margin: 0; font-size: 0.9rem;'>Only {facts['least_favorite']['count']} times</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main() -> None:
    """Main application entry point."""
    setup_page_config()
    st.title("Strong App Analysis")
    
    # Constants
    FILE_PATH = "/Users/nehiljain/Downloads/Strong-2024.csv"
    
    try:
        # Load and prepare data
        workout_data = load_workout_data(FILE_PATH)
        
        # Year selection
        selected_year = st.selectbox(
            "Select Year",
            options=workout_data.available_years,
            index=len(workout_data.available_years) - 1,
            help="Choose a year to analyze workout data"
        )
        
        # Filter data for selected year
        date_spine, year_data = filter_data_by_year(workout_data, selected_year)
        
        if len(year_data) == 0:
            st.warning(f"No workout data found for {selected_year}")
            return
        
        # Display overview
        display_data_overview(year_data, selected_year)
        
        # Frequency visualizations
        monthly_fig = create_monthly_frequency_chart(date_spine)
        st.plotly_chart(monthly_fig, use_container_width=True)
        
        weekday_fig = create_weekday_frequency_chart(date_spine)
        st.plotly_chart(weekday_fig, use_container_width=True)
        
        heatmap_fig = create_activity_heatmap(date_spine)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # One Rep Max Analysis
        st.header("One Rep Max (1RM) Analysis")
        
        exercises_with_1rm = year_data[
            year_data['one_rep_max'].notna()
        ]['Exercise Name'].unique()
        exercises_with_1rm.sort()
        
        if len(exercises_with_1rm) > 0:
            # Single exercise analysis
            selected_exercise = st.selectbox(
                "Select Exercise for Weekly Progress",
                options=exercises_with_1rm,
                help="Choose an exercise to view its One Rep Max progression"
            )
            
            if selected_exercise:
                current_1rm, best_1rm = get_exercise_stats(year_data, selected_exercise)
                display_exercise_analysis(year_data, selected_exercise, current_1rm, best_1rm)
            
            # Multi-exercise comparison
            st.header("Compare Multiple Exercises")
            selected_exercises = st.multiselect(
                "Select Exercises to Compare",
                options=exercises_with_1rm,
                default=[exercises_with_1rm[0]] if len(exercises_with_1rm) > 0 else None,
                help="Choose multiple exercises to compare their One Rep Max progression"
            )
            
            if selected_exercises:
                multi_fig = create_exercise_progress_chart(year_data, selected_exercises)
                st.plotly_chart(multi_fig, use_container_width=True)
        else:
            st.warning("No exercises with weight and reps data found for the selected year")
        
        # Fun Facts Section
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        display_fun_facts(year_data)
    
    except Exception as e:
        st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    main()
