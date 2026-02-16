"""
Visualization Tool
Creates charts and analytics graphs from data and saves them locally
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from langchain_core.tools import tool

# Import config
from config import BASE_DIR

matplotlib.use('Agg')  # Non-interactive backend for file saving

# Output directory for charts
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


@tool
def create_chart(
    data: str,
    chart_type: Literal["bar", "line", "pie", "scatter", "histogram"] = "bar",
    title: str = "Chart",
    x_label: str = "",
    y_label: str = "",
    filename: str = ""
) -> str:
    """
    Create a chart from data and save it to a file.
    
    This tool takes data (typically from SQL query results) and creates 
    visualizations like bar charts, line charts, pie charts, etc.
    
    Args:
        data: JSON string containing the data. Should be a list of dictionaries.
              Example: '[{"month": "Jan", "sales": 1000}, {"month": "Feb", "sales": 1500}]'
              Or dict format: '{"Jan": 1000, "Feb": 1500}'
        
        chart_type: Type of chart to create. Options:
            - "bar": Vertical bar chart (good for comparing categories)
            - "line": Line chart (good for trends over time)
            - "pie": Pie chart (good for showing proportions)
            - "scatter": Scatter plot (good for relationships)
            - "histogram": Histogram (good for distributions)
        
        title: Title for the chart
        
        x_label: Label for x-axis (optional)
        
        y_label: Label for y-axis (optional)
        
        filename: Custom filename (optional, auto-generated if not provided)
    
    Returns:
        Path to the saved chart file
    
    Examples:
        # Bar chart from SQL results
        create_chart(
            data='[{"customer": "Acme Corp", "revenue": 50000}, {"customer": "TechCo", "revenue": 75000}]',
            chart_type="bar",
            title="Top Customers by Revenue",
            x_label="Customer",
            y_label="Revenue ($)"
        )
        
        # Line chart for trends
        create_chart(
            data='[{"month": "Jan", "sales": 10000}, {"month": "Feb", "sales": 12000}]',
            chart_type="line",
            title="Monthly Sales Trend",
            x_label="Month",
            y_label="Sales ($)"
        )
        
        # Pie chart for proportions
        create_chart(
            data='{"North": 30, "South": 25, "East": 25, "West": 20}',
            chart_type="pie",
            title="Sales by Region"
        )
    """
    
    try:
        # Parse data
        data_parsed = json.loads(data)
        
        # Convert to pandas DataFrame for easier manipulation
        if isinstance(data_parsed, list):
            df = pd.DataFrame(data_parsed)
        elif isinstance(data_parsed, dict):
            # Convert dict to list of dicts
            df = pd.DataFrame([data_parsed]).T.reset_index()
            df.columns = ['category', 'value']
        else:
            return f"Error: Data must be a JSON list or dict, got {type(data_parsed)}"
        
        if df.empty:
            return "Error: No data provided"
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{timestamp}.png"
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        filepath = CHARTS_DIR / filename
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create chart based on type
        if chart_type == "bar":
            # Get first two columns (or first column if only one)
            if len(df.columns) >= 2:
                x_data = df.iloc[:, 0]
                y_data = df.iloc[:, 1]
            else:
                x_data = df.index
                y_data = df.iloc[:, 0]
            
            ax.bar(x_data, y_data, color='steelblue')
            ax.set_xlabel(x_label or df.columns[0] if len(df.columns) >= 1 else "Category")
            ax.set_ylabel(y_label or df.columns[1] if len(df.columns) >= 2 else "Value")
            
            # Rotate x-axis labels if they're long
            plt.xticks(rotation=45, ha='right')
        
        elif chart_type == "line":
            if len(df.columns) >= 2:
                x_data = df.iloc[:, 0]
                y_data = df.iloc[:, 1]
            else:
                x_data = df.index
                y_data = df.iloc[:, 0]
            
            ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=8, color='steelblue')
            ax.set_xlabel(x_label or df.columns[0] if len(df.columns) >= 1 else "X")
            ax.set_ylabel(y_label or df.columns[1] if len(df.columns) >= 2 else "Y")
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
        
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                labels = df.iloc[:, 0]
                sizes = df.iloc[:, 1]
            else:
                labels = df.index
                sizes = df.iloc[:, 0]
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures circular pie
        
        elif chart_type == "scatter":
            if len(df.columns) < 2:
                return "Error: Scatter plot requires at least 2 columns of data"
            
            x_data = df.iloc[:, 0]
            y_data = df.iloc[:, 1]
            
            ax.scatter(x_data, y_data, alpha=0.6, s=100, color='steelblue')
            ax.set_xlabel(x_label or df.columns[0])
            ax.set_ylabel(y_label or df.columns[1])
            ax.grid(True, alpha=0.3)
        
        elif chart_type == "histogram":
            data_values = df.iloc[:, -1]  # Use last column
            
            ax.hist(data_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(x_label or df.columns[-1])
            ax.set_ylabel(y_label or "Frequency")
            ax.grid(True, alpha=0.3, axis='y')
        
        else:
            return f"Error: Unknown chart type '{chart_type}'"
        
        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        

        return f"Chart saved successfully to: {filepath}\n\nYou can view it at: {filepath.absolute()}"
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data. {str(e)}\nData received: {data[:100]}..."
    
    except Exception as e:
        return f"Error creating chart: {str(e)}"


@tool
def create_multi_series_chart(
    data: str,
    chart_type: Literal["bar", "line"] = "line",
    title: str = "Multi-Series Chart",
    x_label: str = "",
    y_label: str = "",
    filename: str = ""
) -> str:
    """
    Create a chart with multiple data series (multiple lines or grouped bars).
    
    Useful for comparing multiple metrics over time or across categories.
    
    Args:
        data: JSON string with multiple series. Format:
              '[{"month": "Jan", "sales": 1000, "costs": 800}, 
                {"month": "Feb", "sales": 1500, "costs": 900}]'
              
              First column is treated as x-axis, remaining columns as separate series.
        
        chart_type: "line" for multi-line chart, "bar" for grouped bar chart
        
        title: Chart title
        
        x_label: X-axis label
        
        y_label: Y-axis label
        
        filename: Custom filename (optional)
    
    Returns:
        Path to saved chart
    
    Example:
        create_multi_series_chart(
            data='[{"month": "Jan", "revenue": 10000, "costs": 7000, "profit": 3000},
                   {"month": "Feb", "revenue": 12000, "costs": 7500, "profit": 4500}]',
            chart_type="line",
            title="Revenue, Costs, and Profit Trends",
            x_label="Month",
            y_label="Amount ($)"
        )
    """
    
    try:
        # Parse data
        data_parsed = json.loads(data)
        df = pd.DataFrame(data_parsed)
        
        if df.empty or len(df.columns) < 2:
            return "Error: Need at least 2 columns (x-axis + at least 1 data series)"
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_{chart_type}_{timestamp}.png"
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        filepath = CHARTS_DIR / filename
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # X-axis data (first column)
        x_data = df.iloc[:, 0]
        
        # Plot each series (columns after the first)
        if chart_type == "line":
            for col in df.columns[1:]:
                ax.plot(x_data, df[col], marker='o', linewidth=2, markersize=6, label=col)
            
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
        
        elif chart_type == "bar":
            # Grouped bar chart
            x_pos = range(len(x_data))
            width = 0.8 / (len(df.columns) - 1)  # Width of each bar
            
            for i, col in enumerate(df.columns[1:]):
                offset = width * i - (width * (len(df.columns) - 2) / 2)
                ax.bar([x + offset for x in x_pos], df[col], width, label=col)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_data, rotation=45, ha='right')
            ax.legend(loc='best')
        
        ax.set_xlabel(x_label or df.columns[0])
        ax.set_ylabel(y_label or "Value")
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return f"Multi-series chart saved to: {filepath}\n\nView at: {filepath.absolute()}"
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data. {str(e)}"
    
    except Exception as e:
        return f"Error creating multi-series chart: {str(e)}"

