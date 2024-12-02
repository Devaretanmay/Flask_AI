from flask import Flask, jsonify, Response, request, redirect, url_for, render_template
import flask
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from vanna.remote import VannaDefault
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Vanna
v = VannaDefault(
    api_key=os.getenv('VANNA_API_KEY'),
    model=os.getenv('VANNA_MODEL', 'tanmay_model')
)

# Setup database connection
db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(db_url)

# Simple in-memory cache
class Cache:
    def __init__(self):
        self.data = {}
    
    def set(self, id, field, value):
        if id not in self.data:
            self.data[id] = {}
        self.data[id][field] = value
    
    def get(self, id, field):
        return self.data.get(id, {}).get(field)

cache = Cache()

# Pre-written queries dictionary
PREDEFINED_QUERIES = {
    "Daily Energy Consumption": {
        "sql": """
            SELECT 
                date_trunc('hour', tr.date_time) as hour,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.id
            WHERE tr.date_time >= NOW() - INTERVAL '7 days'
            AND t.tag_name LIKE '%ENERGY_CONSUMTION%'
            GROUP BY hour
            ORDER BY hour;
        """,
        "description": "Shows hourly energy consumption patterns over the last 7 days",
        "visualization": "line"
    },
    
    "Weekday vs Weekend Analysis": {
        "sql": """
            SELECT 
                CASE 
                    WHEN EXTRACT(DOW FROM tr.date_time) IN (0, 6) THEN 'Weekend'
                    ELSE 'Weekday'
                END as day_type,
                EXTRACT(HOUR FROM tr.date_time) as hour_of_day,
                AVG(tr.tag_value) as avg_consumption
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.id
            WHERE t.tag_name LIKE '%ENERGY_CONSUMTION%'
            GROUP BY day_type, hour_of_day
            ORDER BY hour_of_day;
        """,
        "description": "Compares energy usage patterns between weekdays and weekends",
        "visualization": "bar"
    },
    
    "Peak Hours Analysis": {
        "sql": """
            SELECT 
                EXTRACT(HOUR FROM tr.date_time) as hour_of_day,
                AVG(tr.tag_value) as avg_consumption,
                COUNT(*) as reading_count
            FROM tag_readings tr
            JOIN tags t ON tr.tag_id = t.id
            WHERE t.tag_name LIKE '%ENERGY_CONSUMTION%'
            GROUP BY hour_of_day
            ORDER BY avg_consumption DESC;
        """,
        "description": "Identifies hours with highest energy consumption",
        "visualization": "bar"
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/v0/execute_query', methods=['POST'])
def execute_query():
    try:
        data = request.json
        query_text = data.get('query', '')
        is_predefined = data.get('is_predefined', False)
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
            
        if is_predefined:
            # Handle pre-written query
            if query_text not in PREDEFINED_QUERIES:
                return jsonify({'error': 'Invalid pre-written query'}), 400
                
            query_info = PREDEFINED_QUERIES[query_text]
            sql = query_info["sql"]
            viz_type = query_info["visualization"]
        else:
            # Use Vanna AI for custom queries
            try:
                sql = v.generate_sql(query_text)
                viz_type = "auto"  # Let visualization function decide
            except Exception as e:
                logger.error(f"Vanna API error: {str(e)}")
                return jsonify({
                    'error': 'Custom query generation failed. Please try one of the predefined queries:',
                    'available_queries': list(PREDEFINED_QUERIES.keys())
                }), 429

        # Execute query and get results
        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql), connection)
                
            if df.empty:
                return jsonify({'error': 'No data found for the query'}), 404
                
            # Generate cache ID and store data
            cache_id = str(uuid.uuid4())
            cache.set(id=cache_id, field='df', value=df)
            cache.set(id=cache_id, field='question', value=query_text)
            cache.set(id=cache_id, field='sql', value=sql)
            cache.set(id=cache_id, field='viz_type', value=viz_type)
            
            # Convert DataFrame to list of dictionaries for JSON response
            records = df.to_dict(orient='records')
            columns = [{"title": col, "field": col} for col in df.columns]
            
            return jsonify({
                "type": "sql_result",
                "sql": sql,
                "data": {
                    "records": records,
                    "columns": columns,
                    "total_rows": len(df)
                },
                "id": cache_id,
                "viz_type": viz_type
            })
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v0/get_recommended_queries', methods=['GET'])
def get_recommended_queries():
    """Get AI-recommended queries based on recent usage patterns"""
    try:
        # Initial recommended queries
        recommended_queries = [
            {
                "query": "How does the current week's consumption compare to last week?",
                "category": "Time Comparison",
                "description": "Weekly consumption trend analysis"
            },
            {
                "query": "Show me any unusual patterns in the last 24 hours",
                "category": "Anomaly Detection",
                "description": "Identify abnormal consumption patterns"
            },
            {
                "query": "What are the peak consumption hours today?",
                "category": "Usage Pattern",
                "description": "Daily peak usage analysis"
            },
            {
                "query": "Compare energy usage between different zones",
                "category": "Location Analysis",
                "description": "Regional consumption comparison"
            },
            {
                "query": "Show me the most efficient devices",
                "category": "Device Analysis",
                "description": "Device efficiency ranking"
            }
        ]
        
        return jsonify({
            'queries': recommended_queries,
            'total': len(recommended_queries)
        })
        
    except Exception as e:
        logger.error(f"Error getting recommended queries: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v0/get_predefined_queries', methods=['GET'])
def get_predefined_queries():
    """Get list of available predefined queries"""
    try:
        queries = []
        for name, info in PREDEFINED_QUERIES.items():
            queries.append({
                'name': name,
                'description': info['description'],
                'category': 'Energy Analysis'
            })
        return jsonify({
            'queries': queries,
            'total': len(queries)
        })
    except Exception as e:
        logger.error(f"Error getting predefined queries: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v0/generate_plotly_figure', methods=['GET'])
def generate_plotly_figure():
    try:
        id = request.args.get('id')
        df = pd.DataFrame(cache.get(id=id, field='df'))
        question = cache.get(id=id, field='question')
        sql = cache.get(id=id, field='sql')
        viz_type = cache.get(id=id, field='viz_type')
        
        if df is None or df.empty:
            return jsonify({"error": "No data found for this query"}), 404
            
        fig = generate_visualization(df, viz_type)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        # Convert figure to dict and handle numpy types
        fig_dict = {
            "data": [convert_numpy(trace.to_plotly_json()) for trace in fig.data],
            "layout": convert_numpy(fig.layout.to_plotly_json())
        }
        
        return jsonify(fig_dict)
        
    except Exception as e:
        logger.error(f"Error generating figure: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_visualization(df, viz_type="auto"):
    """Generate appropriate visualization based on data characteristics"""
    try:
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        # Time series data
        time_columns = ['hour', 'date_time', 'period_start', 'period_end']
        if viz_type == "line" or (viz_type == "auto" and any(col in df.columns for col in time_columns)):
            time_col = next(col for col in time_columns if col in df.columns)
            value_cols = [col for col in df.columns if any(x in col.lower() for x in ['consumption', 'value', 'reading', 'count'])]
            
            fig = go.Figure()
            for col in value_cols:
                fig.add_trace(go.Scatter(
                    x=df[time_col], 
                    y=df[col], 
                    name=col.replace('_', ' ').title(),
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            fig.update_layout(
                title="Time Series Analysis",
                xaxis_title=time_col.replace('_', ' ').title(),
                yaxis_title="Value"
            )

        # Bar charts for categorical data
        elif viz_type == "bar" or (viz_type == "auto" and any(col in df.columns for col in ['day_type', 'region', 'device_name', 'hour_of_day'])):
            cat_col = next(col for col in ['day_type', 'region', 'device_name', 'hour_of_day'] if col in df.columns)
            value_cols = [col for col in df.columns if any(x in col.lower() for x in ['consumption', 'value', 'reading', 'count'])]
            
            fig = go.Figure()
            for col in value_cols:
                fig.add_trace(go.Bar(
                    x=df[cat_col], 
                    y=df[col], 
                    name=col.replace('_', ' ').title()
                ))
            fig.update_layout(
                title=f"{cat_col.replace('_', ' ').title()} Analysis",
                xaxis_title=cat_col.replace('_', ' ').title(),
                yaxis_title="Value",
                barmode='group'
            )

        # Default to bar chart for other numeric data
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig = go.Figure(data=go.Bar(
                    x=df[numeric_cols[0]], 
                    y=df[numeric_cols[1]],
                    name=numeric_cols[1].replace('_', ' ').title()
                ))
                fig.update_layout(
                    title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                    xaxis_title=numeric_cols[0].replace('_', ' ').title(),
                    yaxis_title=numeric_cols[1].replace('_', ' ').title()
                )
            else:
                fig = go.Figure(data=go.Histogram(
                    x=df[numeric_cols[0]],
                    name=numeric_cols[0].replace('_', ' ').title()
                ))
                fig.update_layout(
                    title=f"Distribution of {numeric_cols[0]}",
                    xaxis_title=numeric_cols[0].replace('_', ' ').title(),
                    yaxis_title="Count"
                )

        # Update layout with light theme and better formatting
        fig.update_layout(
            template="plotly",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            margin=dict(t=50, l=50, r=50, b=50),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#dee2e6',
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='#e9ecef',
                zerolinecolor='#dee2e6',
                zerolinewidth=1
            ),
            yaxis=dict(
                gridcolor='#e9ecef',
                zerolinecolor='#dee2e6',
                zerolinewidth=1
            )
        )
        
        return fig

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

if __name__ == '__main__':
    app.run(debug=True)
