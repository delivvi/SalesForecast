import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import socket
from werkzeug.serving import make_server
from plotly.subplots import make_subplots

class SalesDashboard:
    def __init__(self):
        """Инициализация панели мониторинга продаж"""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.app.title = "Coffee Sales Analysis & Forecasting"
        self.sales_data = None
        self.forecast_data = None
        self.feature_importance = None
        self.server = None  # Сохранить экземпляр сервера для выключения
        self._create_layout()
        self._register_callbacks()
    
    def _create_layout(self):
        """Создание макета панели мониторинга"""
        self.app.layout = dbc.Container([
            html.H1("Sales Analysis & Forecasting Dashboard", className="my-4"),
            
            # Фильтры
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="date-picker-range",
                                display_format="YYYY-MM-DD"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Product Category"),
                            dcc.Dropdown(
                                id="category-dropdown",
                                placeholder="Select categories"
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Time Aggregation"),
                            dcc.RadioItems(
                                id="time-aggregation",
                                options=[
                                    {"label": "Daily", "value": "D"},
                                    {"label": "Weekly", "value": "W"},
                                    {"label": "Monthly", "value": "M"},
                                    {"label": "Quarterly", "value": "Q"}
                                ],
                                value="D",
                                inline=True
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Apply Filters", id="apply-filter-button", color="primary", className="me-2 mt-4"),
                            dbc.Button("Reset", id="reset-button", color="secondary", className="mt-4")
                        ], width=2)
                    ])
                ])
            ], className="mb-4"),
            
            # Вкладки для различных панелей мониторинга
            dbc.Tabs([
                # Вкладка 1: Обзор продаж
                dbc.Tab(label="Sales Overview", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Total Revenue", className="card-title"),
                                    html.H3(id="total-sales", className="card-text text-primary")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Average Check", className="card-title"),
                                    html.H3(id="avg-check", className="card-text text-success")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Total Transactions", className="card-title"),
                                    html.H3(id="total-transactions", className="card-text text-info")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("vs Previous Period", className="card-title"),
                                    html.H3(id="sales-growth", className="card-text text-warning")
                                ])
                            ])
                        ], width=3)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Sales Dynamics Over Time"),
                                dbc.CardBody([
                                    dcc.Graph(id="sales-trend-graph")
                                ])
                            ])
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Sales by Category"),
                                dbc.CardBody([
                                    dcc.Graph(id="sales-by-category")
                                ])
                            ])
                        ], width=4)
                    ], className="mb-4")
                ]),
                
                # Вкладка 2: Детальный анализ продуктов
                dbc.Tab(label="Product Analysis", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Top Products by Sales"),
                                dbc.CardBody([
                                    dcc.Graph(id="top-products-graph")
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("ABC Analysis"),
                                dbc.CardBody([
                                    dcc.Graph(id="abc-analysis-graph")
                                ])
                            ])
                        ], width=6)
                    ], className="mb-4")
                ]),
                
                # Вкладка 3: Прогнозирование продаж
                dbc.Tab(label="Sales Forecasting", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Sales Forecast"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Forecast Period (Days)"),
                                            dcc.Slider(
                                                id="forecast-days-slider",
                                                min=7,
                                                max=90,
                                                step=7,
                                                value=30,
                                                marks={i: f"{i}" for i in range(0, 91, 7)}
                                            )
                                        ], width=12)
                                    ], className="mb-3"),
                                    dcc.Graph(id="forecast-graph")
                                ])
                            ])
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Feature Importance"),
                                dbc.CardBody([
                                    dcc.Graph(id="feature-importance-graph")
                                ])
                            ])
                        ], width=4)
                    ])
                ])
            ])
        ], fluid=True)
    
    def _register_callbacks(self):
        """Регистрация всех обратных вызовов для панели мониторинга"""
        
        # Set initial date picker range based on data
        @self.app.callback(
            [
                Output("date-picker-range", "min_date_allowed"),
                Output("date-picker-range", "max_date_allowed"),
                Output("date-picker-range", "start_date"),
                Output("date-picker-range", "end_date")
            ],
            [Input("apply-filter-button", "n_clicks")]
        )
        def update_date_picker(n_clicks):
            if self.sales_data is None or len(self.sales_data) == 0:
                today = datetime.now().date()
                min_date = today - timedelta(days=30)
                return min_date, today, min_date, today
            
            # Get min and max dates from data
            min_date = self.sales_data['sale_date'].min().date()
            max_date = self.sales_data['sale_date'].max().date()
            
            return min_date, max_date, min_date, max_date
        
        # Update category dropdown options to show coffee types
        @self.app.callback(
            Output("category-dropdown", "options"),
            [Input("apply-filter-button", "n_clicks")]
        )
        def update_category_options(n_clicks):
            if self.sales_data is None:
                return []
            
            # Check if product_name column exists (for coffee products)
            if 'product_name' in self.sales_data.columns:
                unique_products = sorted(self.sales_data['product_name'].unique())
                options = [{'label': 'All Coffee Types', 'value': 'all'}]
                options.extend([{'label': product, 'value': product} for product in unique_products if product])
                return options
            # Fallback to coffee_name if it exists
            elif 'coffee_name' in self.sales_data.columns:
                unique_products = sorted(self.sales_data['coffee_name'].unique())
                options = [{'label': 'All Coffee Types', 'value': 'all'}]
                options.extend([{'label': product, 'value': product} for product in unique_products if product])
                return options
            # Legacy support for category
            elif 'category' in self.sales_data.columns:
                categories = sorted(self.sales_data['category'].unique())
                return [{'label': cat, 'value': cat} for cat in categories if cat]
            
            return []
        
        @self.app.callback(
            [
                Output("total-sales", "children"),
                Output("avg-check", "children"),
                Output("total-transactions", "children"),
                Output("sales-growth", "children")
            ],
            [
                Input("apply-filter-button", "n_clicks")
            ],
            [
                dash.dependencies.State("date-picker-range", "start_date"),
                dash.dependencies.State("date-picker-range", "end_date"),
                dash.dependencies.State("category-dropdown", "value")
            ]
        )
        def update_summary_stats(n_clicks, start_date, end_date, selected_categories):
            if self.sales_data is None:
                return "$0", "$0", "0", "0%"
            
            # Filter data by date range and category
            filtered_data = self._filter_data(start_date, end_date, selected_categories)
            
            # Use price for coffee sales instead of total_amount
            price_col = 'price' if 'price' in filtered_data.columns else 'total_amount'
            
            # Calculate summary statistics
            total_sales = filtered_data[price_col].sum()
            
            # Calculate average check (average transaction value)
            avg_check = total_sales / len(filtered_data) if len(filtered_data) > 0 else 0
            
            # Count total transactions
            total_transactions = len(filtered_data)
            
            # Calculate sales growth (from first half to second half of the period)
            mid_point = len(filtered_data) // 2
            if mid_point > 0:
                first_half = filtered_data.iloc[:mid_point][price_col].sum()
                second_half = filtered_data.iloc[mid_point:][price_col].sum()
                if first_half > 0:
                    growth = ((second_half - first_half) / first_half) * 100
                else:
                    growth = 0
            else:
                growth = 0
            
            return (
                f"${total_sales:,.2f}",
                f"${avg_check:,.2f}",
                f"{total_transactions:,}",
                f"{growth:+.1f}%"
            )
        
        @self.app.callback(
            Output("sales-trend-graph", "figure"),
            [
                Input("apply-filter-button", "n_clicks"),
                Input("reset-button", "n_clicks"),
                Input("time-aggregation", "value")
            ],
            [
                dash.dependencies.State("date-picker-range", "start_date"),
                dash.dependencies.State("date-picker-range", "end_date"),
                dash.dependencies.State("category-dropdown", "value")
            ]
        )
        def update_sales_trend(apply_clicks, reset_clicks, time_agg, start_date, end_date, selected_categories):
            if self.sales_data is None:
                return go.Figure()
            
            # Filter data by date range and category
            filtered_data = self._filter_data(start_date, end_date, selected_categories)
            
            # Use price for coffee sales instead of total_amount
            price_col = 'price' if 'price' in filtered_data.columns else 'total_amount'
            
            # Resample based on time aggregation
            agg_map = {
                'D': 'Day',
                'W': 'Week',
                'M': 'Month',
                'Q': 'Quarter'
            }
            
            # Group by date and aggregate price
            sales_agg = filtered_data.resample(time_agg, on='sale_date')[price_col].sum().reset_index()
            
            # Create the figure
            fig = px.line(
                sales_agg,
                x='sale_date',
                y=price_col,
                title=f'Sales by {agg_map[time_agg]}',
                labels={'sale_date': 'Date', price_col: 'Sales Amount'}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Sales Amount ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output("sales-by-category", "figure"),
            [
                Input("apply-filter-button", "n_clicks")
            ],
            [
                dash.dependencies.State("date-picker-range", "start_date"),
                dash.dependencies.State("date-picker-range", "end_date")
            ]
        )
        def update_sales_by_category(n_clicks, start_date, end_date):
            if self.sales_data is None:
                # Create a placeholder figure
                fig = go.Figure()
                fig.update_layout(
                    title="No product data available",
                    annotations=[
                        dict(
                            text="No data available for product categories",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Filter data by date range
            filtered_data = self._filter_data(start_date, end_date)
            
            # Use price for coffee sales instead of total_amount
            price_col = 'price' if 'price' in filtered_data.columns else 'total_amount'
            
            # Determine which column to use for product type
            product_col = None
            if 'product_name' in filtered_data.columns:
                product_col = 'product_name'
            elif 'coffee_name' in filtered_data.columns:
                product_col = 'coffee_name'
            elif 'category' in filtered_data.columns:
                product_col = 'category'
            else:
                # No suitable column found
                fig = go.Figure()
                fig.update_layout(
                    title="No product categories available",
                    annotations=[
                        dict(
                            text="No product category data found",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Group by product type and sum sales
            category_sales = filtered_data.groupby(product_col)[price_col].sum().reset_index()
            
            # Create the figure
            fig = px.pie(
                category_sales,
                values=price_col,
                names=product_col,
                title='Sales by Coffee Type'
            )
            
            # Update layout
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
        
        @self.app.callback(
            Output("forecast-graph", "figure"),
            [
                Input("forecast-days-slider", "value")
            ]
        )
        def update_forecast_graph(forecast_days):
            if self.sales_data is None or self.forecast_data is None:
                # Create a placeholder figure
                fig = go.Figure()
                fig.update_layout(
                    title="No forecast data available",
                    annotations=[
                        dict(
                            text="Please generate a forecast first",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Filter forecast data to the specified number of days
            forecast = self.forecast_data.iloc[:forecast_days]
            
            # Create the figure
            fig = go.Figure()
            
            # Determine which column to use for price in historical data
            price_col = 'price' if 'price' in self.sales_data.columns else 'total_amount'
            
            # Determine which column to use for predictions in forecast data
            forecast_col = None
            for possible_col in ['predicted_amount', 'predicted', 'prediction', 'forecast', 'price_pred']:
                if possible_col in forecast.columns:
                    forecast_col = possible_col
                    break
            
            if forecast_col is None:
                # If no standard prediction column is found, use the first numeric column that isn't sale_date
                for col in forecast.columns:
                    if col != 'sale_date' and pd.api.types.is_numeric_dtype(forecast[col]):
                        forecast_col = col
                        break
            
            if forecast_col is None:
                # Still no suitable column found
                fig.update_layout(
                    title="Forecast data format not recognized",
                    annotations=[
                        dict(
                            text="Cannot find prediction values in forecast data",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Show ALL historical data, not just the last 90 days
            if len(self.sales_data) > 0:
                # Make sure data is sorted by date
                historical_data = self.sales_data.sort_values('sale_date')
                
                # Group by date if there are multiple entries per date
                if len(historical_data) > len(historical_data['sale_date'].unique()):
                    historical_data = historical_data.groupby('sale_date')[price_col].sum().reset_index()
                
                # Add historical line
                fig.add_trace(
                    go.Scatter(
                        x=historical_data['sale_date'],
                        y=historical_data[price_col],
                        mode='lines',
                        name='Actual Sales',
                        line=dict(color='blue', width=2)
                    )
                )
                
                # Fill area under historical line
                fig.add_trace(
                    go.Scatter(
                        x=historical_data['sale_date'],
                        y=historical_data[price_col],
                        mode='none',
                        name='Historical Data',
                        fill='tozeroy',
                        fillcolor='rgba(0, 0, 255, 0.1)',
                        showlegend=False
                    )
                )
            
            # Add forecast data
            fig.add_trace(
                go.Scatter(
                    x=forecast['sale_date'],
                    y=forecast[forecast_col],
                    mode='lines',
                    name="Forecast",
                    line=dict(color='red', width=2)
                )
            )
            
            # Fill area under forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast['sale_date'],
                    y=forecast[forecast_col],
                    mode='none',
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    showlegend=False
                )
            )
            
            # Add confidence interval if available
            if 'lower_bound' in forecast.columns and 'upper_bound' in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['sale_date'].tolist() + forecast['sale_date'].tolist()[::-1],
                        y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        name='95% Confidence Interval'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f'Historical Sales and Forecast for Next {forecast_days} Days',
                xaxis_title='Date',
                yaxis_title='Sales Amount ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output("feature-importance-graph", "figure"),
            [
                Input("apply-filter-button", "n_clicks")
            ]
        )
        def update_feature_importance(n_clicks):
            if self.feature_importance is None:
                # Create a placeholder figure
                fig = go.Figure()
                fig.update_layout(
                    title="No feature importance data available",
                    annotations=[
                        dict(
                            text="Train a model first to see feature importance",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Create the figure
            fig = px.bar(
                self.feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Importance',
                yaxis_title='Feature',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
        
        @self.app.callback(
            Output("top-products-graph", "figure"),
            [Input("apply-filter-button", "n_clicks")],
            [
                dash.dependencies.State("date-picker-range", "start_date"),
                dash.dependencies.State("date-picker-range", "end_date"),
                dash.dependencies.State("category-dropdown", "value")
            ]
        )
        def update_top_products(n_clicks, start_date, end_date, selected_categories):
            if self.sales_data is None:
                # Create a placeholder figure
                fig = go.Figure()
                fig.update_layout(
                    title="No product data available",
                    annotations=[
                        dict(
                            text="No data available for product analysis",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Filter data
            filtered_data = self._filter_data(start_date, end_date, selected_categories)
            
            # Use price for coffee sales instead of total_amount
            price_col = 'price' if 'price' in filtered_data.columns else 'total_amount'
            
            # Determine which column to use for product type
            product_col = None
            if 'product_name' in filtered_data.columns:
                product_col = 'product_name'
            elif 'coffee_name' in filtered_data.columns:
                product_col = 'coffee_name'
            else:
                # No suitable column found
                fig = go.Figure()
                fig.update_layout(
                    title="No product data available",
                    annotations=[
                        dict(
                            text="No product data found",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Get top 10 products by sales
            top_products = filtered_data.groupby(product_col)[price_col].sum().nlargest(10).reset_index()
            
            # Sort by sales amount
            top_products = top_products.sort_values(price_col)
            
            # Create horizontal bar chart
            fig = px.bar(
                top_products,
                y=product_col,
                x=price_col,
                orientation='h',
                title='Top Coffee Types by Sales',
                labels={product_col: 'Coffee Type', price_col: 'Sales Amount'}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Sales Amount ($)',
                yaxis_title='Coffee Type',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        @self.app.callback(
            Output("abc-analysis-graph", "figure"),
            [Input("apply-filter-button", "n_clicks")],
            [
                dash.dependencies.State("date-picker-range", "start_date"),
                dash.dependencies.State("date-picker-range", "end_date"),
                dash.dependencies.State("category-dropdown", "value")
            ]
        )
        def update_abc_analysis(n_clicks, start_date, end_date, selected_categories):
            if self.sales_data is None:
                # Create a placeholder figure
                fig = go.Figure()
                fig.update_layout(
                    title="No data available for ABC analysis",
                    annotations=[
                        dict(
                            text="No data available for ABC analysis",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Filter data
            filtered_data = self._filter_data(start_date, end_date, selected_categories)
            
            # Use price for coffee sales instead of total_amount
            price_col = 'price' if 'price' in filtered_data.columns else 'total_amount'
            
            # Determine which column to use for product type
            product_col = None
            if 'product_name' in filtered_data.columns:
                product_col = 'product_name'
            elif 'coffee_name' in filtered_data.columns:
                product_col = 'coffee_name'
            else:
                # No suitable column found
                fig = go.Figure()
                fig.update_layout(
                    title="No product data available for ABC analysis",
                    annotations=[
                        dict(
                            text="No product data found for ABC analysis",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5
                        )
                    ]
                )
                return fig
            
            # Group by product and sum sales
            product_sales = filtered_data.groupby(product_col)[price_col].sum().reset_index()
            
            # Get total sales
            total_sales = product_sales[price_col].sum()
            
            # Sort by sales amount
            product_sales = product_sales.sort_values(price_col, ascending=False)
            
            # Calculate cumulative percentage
            product_sales['percent'] = product_sales[price_col] / total_sales * 100
            product_sales['cumulative_percent'] = product_sales['percent'].cumsum()
            
            # Classify products into A, B, C categories
            product_sales['category'] = 'C'
            product_sales.loc[product_sales['cumulative_percent'] <= 70, 'category'] = 'A'
            product_sales.loc[(product_sales['cumulative_percent'] > 70) & (product_sales['cumulative_percent'] <= 90), 'category'] = 'B'
            
            # Count products in each category
            category_counts = product_sales.groupby('category').size().reset_index(name='count')
            category_counts['percent'] = category_counts['count'] / category_counts['count'].sum() * 100
            
            # Calculate sales by category
            category_sales = product_sales.groupby('category')[price_col].sum().reset_index()
            category_sales['percent'] = category_sales[price_col] / total_sales * 100
            
            # Create figure with two subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                subplot_titles=('Revenue Distribution', 'Product Count Distribution')
            )
            
            # Add pie charts
            fig.add_trace(go.Pie(
                labels=category_sales['category'],
                values=category_sales[price_col],
                domain=dict(x=[0, 0.45]),
                title="Revenue by Category",
                hole=0.4,
                marker=dict(
                    colors=['#ff7f0e', '#1f77b4', '#2ca02c']
                )
            ), row=1, col=1)
            
            fig.add_trace(go.Pie(
                labels=category_counts['category'],
                values=category_counts['count'],
                domain=dict(x=[0.55, 1]),
                title="Product Count by Category",
                hole=0.4,
                marker=dict(
                    colors=['#ff7f0e', '#1f77b4', '#2ca02c']
                )
            ), row=1, col=2)
            
            # Update layout
            fig.update_layout(
                title="ABC Analysis (A: Top 70% of Revenue, B: Next 20%, C: Remaining 10%)",
                annotations=[
                    dict(
                        text="Revenue Distribution",
                        x=0.225,
                        y=0.5,
                        font_size=12,
                        showarrow=False
                    ),
                    dict(
                        text="Product Distribution",
                        x=0.775,
                        y=0.5,
                        font_size=12,
                        showarrow=False
                    )
                ]
            )
            
            return fig
    
    def _filter_data(self, start_date, end_date, selected_categories=None):
        """
        Filter the sales data by date range and categories
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            selected_categories: List of selected category names or coffee types
            
        Returns:
            Filtered pandas DataFrame
        """
        if self.sales_data is None:
            return pd.DataFrame()
        
        # Create a copy of the data
        filtered_data = self.sales_data.copy()
        
        # Filter by date if dates are provided
        if start_date and end_date:
            filtered_data = filtered_data[
                (filtered_data['sale_date'] >= pd.to_datetime(start_date)) &
                (filtered_data['sale_date'] <= pd.to_datetime(end_date))
            ]
        
        # Filter by selected categories if provided
        if selected_categories:
            # Skip if 'all' is selected
            if selected_categories != 'all':
                # Determine which column to use for filtering
                if 'product_name' in filtered_data.columns:
                    filter_col = 'product_name'
                elif 'coffee_name' in filtered_data.columns:
                    filter_col = 'coffee_name'
                elif 'category' in filtered_data.columns:
                    filter_col = 'category'
                else:
                    # No suitable column found, return as is
                    return filtered_data
                
                # Apply filtering based on type of selected_categories
                if isinstance(selected_categories, list) and len(selected_categories) > 0:
                    filtered_data = filtered_data[filtered_data[filter_col].isin(selected_categories)]
                elif isinstance(selected_categories, str) and selected_categories != 'all':
                    filtered_data = filtered_data[filtered_data[filter_col] == selected_categories]
        
        return filtered_data
    
    def update_data(self, sales_data, forecast_data=None, feature_importance=None):
        """
        Update the dashboard with new data
        
        Args:
            sales_data: DataFrame with sales data
            forecast_data: DataFrame with forecasted sales
            feature_importance: DataFrame with feature importance
        """
        # Ensure sale_date is datetime
        if sales_data is not None and 'sale_date' in sales_data.columns:
            sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
        
        # Process coffee data: if coffee_name exists but product_name doesn't, copy it
        if sales_data is not None and 'coffee_name' in sales_data.columns and 'product_name' not in sales_data.columns:
            sales_data['product_name'] = sales_data['coffee_name']
        
        self.sales_data = sales_data
        self.forecast_data = forecast_data
        self.feature_importance = feature_importance
    
    def find_available_port(self, start_port=8050, max_port=8100):
        """
        Find an available port for the dashboard server
        
        Args:
            start_port: Port to start searching from
            max_port: Maximum port to try
            
        Returns:
            Available port number or None if no port is available
        """
        for port in range(start_port, max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return port
                except socket.error:
                    continue
        return None
    
    def run_server(self, debug=False, port=8050):
        """Run the Dash server"""
        try:
            # Find available port if the specified one is not available
            if port == 8050:  # Only auto-detect if using default port
                available_port = self.find_available_port(start_port=port)
                if available_port and available_port != port:
                    print(f"Port {port} is not available, using port {available_port} instead")
                    port = available_port
            
            # Store server reference for shutdown
            self.server = make_server('0.0.0.0', port, self.app.server)
            self.server.serve_forever()
            return True
        except Exception as e:
            print(f"Error starting dashboard server: {str(e)}")
            return False

    def shutdown(self):
        """Shutdown the dashboard server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            print("Dashboard server shutdown")
        else:
            print("Dashboard server is not running") 