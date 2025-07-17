"""
Results display components for portfolio simulator.
"""

import streamlit as st
import io
import pandas as pd
from typing import Dict, Any
from reports import PDFReportGenerator, ReportFactory


class ResultsDisplay:
    """Handles display of simulation results, charts, and metrics."""
    
    def __init__(self, explanations: Dict[str, str]):
        """Initialize with explanations for tooltips."""
        self.explanations = explanations
    
    def render_all_results(self, session_data: Dict[str, Any]):
        """Render all results sections."""
        # Display optimized portfolio allocation if available
        self._render_optimized_allocation(session_data)
        
        # Display simulation results
        self._render_simulation_metrics(session_data['results'])
        
        # Plot simulation distribution
        self._render_distribution_chart(session_data['fig_dist'])
        
        # Historical performance plot
        self._render_historical_performance(session_data['fig_hist'])
        
        # Historical drawdown plot
        self._render_drawdown_analysis(session_data['fig_dd'])
        
        # Weight drift plot
        self._render_weight_drift_analysis(session_data['fig_drift'])
        
        # Backtesting results
        self._render_backtest_results(session_data['backtest_results'])
        
        # PDF and CSV download buttons
        self._render_download_buttons(session_data)
    
    def _render_simulation_metrics(self, results: Dict[str, float]):
        """Render simulation results metrics."""
        st.header('Simulation Results')
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Historical metrics
        with col1:
            self._render_metric(
                'Historical Annual Return', 
                f"{results['Historical Annual Return']:.2%}"
            )
            self._render_metric(
                'Historical Annual Volatility', 
                f"{results['Historical Annual Volatility']:.2%}"
            )
            self._render_metric(
                'Historical Sharpe Ratio', 
                f"{results['Historical Sharpe Ratio']:.2f}"
            )
            self._render_metric(
                'Historical Sortino Ratio', 
                f"{results['Historical Sortino Ratio']:.2f}"
            )
            self._render_metric(
                'Historical Max Drawdown', 
                f"{results['Historical Max Drawdown']:.2%}"
            )
        
        # Column 2: Simulated final values
        with col2:
            self._render_metric(
                'Mean Final Value (Inflation-Adjusted, DCA)', 
                f"€{results['Mean Final Value (Inflation-Adjusted, DCA)']:.2f}"
            )
            self._render_metric(
                'Median Final Value (Inflation-Adjusted, DCA)', 
                f"€{results['Median Final Value (Inflation-Adjusted, DCA)']:.2f}"
            )
            self._render_metric(
                'Mean Final Value (Lump-Sum Comparison)', 
                f"€{results['Mean Final Value (Lump-Sum Comparison)']:.2f}"
            )
        
        # Column 3: Risk metrics
        with col3:
            self._render_metric(
                'Std Dev of Final Values (DCA)', 
                f"€{results['Std Dev of Final Values (DCA)']:.2f}"
            )
            self._render_metric(
                '95% VaR (Absolute Loss, DCA)', 
                f"€{results['95% VaR (Absolute Loss, DCA)']:.2f}"
            )
            self._render_metric(
                '95% CVaR (Absolute Loss, DCA)', 
                f"€{results['95% CVaR (Absolute Loss, DCA)']:.2f}"
            )
            self._render_metric(
                'Effective Cost Drag (%)', 
                f"{results['Effective Cost Drag (%)']:.2f}%"
            )
    
    def _render_metric(self, key: str, value: str):
        """Render a single metric with tooltip."""
        help_text = self.explanations.get(key, "")
        st.metric(key, value, help=help_text)
    
    def _render_distribution_chart(self, fig_dist):
        """Render simulation distribution chart."""
        st.header('Distribution of Outcomes')
        st.pyplot(fig_dist)
    
    def _render_historical_performance(self, fig_hist):
        """Render historical performance chart."""
        st.header('Historical Performance')
        st.plotly_chart(fig_hist)
    
    def _render_drawdown_analysis(self, fig_dd):
        """Render drawdown analysis chart."""
        st.header('Historical Drawdown')
        st.plotly_chart(fig_dd)
    
    def _render_weight_drift_analysis(self, fig_drift):
        """Render weight drift analysis chart."""
        st.header('Weight Drift Analysis')
        st.plotly_chart(fig_drift)
    
    def _render_backtest_results(self, backtest_results: Dict[str, float]):
        """Render backtesting results metrics."""
        st.header('Backtesting Results')
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_metric(
                'Total Historical Return (DCA)', 
                f"{backtest_results['Total Return (DCA)']:.2%}"
            )
            self._render_metric(
                'Total Historical Return (Lump-Sum)', 
                f"{backtest_results['Total Return (Lump-Sum)']:.2%}"
            )
            self._render_metric(
                'Annualized Return', 
                f"{backtest_results['Annualized Return']:.2%}"
            )
        
        with col2:
            self._render_metric(
                'Annualized Volatility', 
                f"{backtest_results['Annualized Volatility']:.2%}"
            )
            self._render_metric(
                'Sharpe Ratio', 
                f"{backtest_results['Sharpe Ratio']:.2f}"
            )
            self._render_metric(
                'Sortino Ratio', 
                f"{backtest_results['Sortino Ratio']:.2f}"
            )
            self._render_metric(
                'Max Drawdown', 
                f"{backtest_results['Max Drawdown']:.2%}"
            )
    
    def _render_download_buttons(self, session_data: Dict[str, Any]):
        """Render PDF and CSV download buttons."""
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_pdf_download_button(session_data)
        
        with col2:
            self._render_csv_download_button(session_data['results'])
    
    def _render_pdf_download_button(self, session_data: Dict[str, Any]):
        """Render PDF report download button."""
        if st.button('Generate PDF Report'):
            # Create charts dictionary
            charts = ReportFactory.create_charts_dict(
                fig_pie_original=session_data['fig_pie_original'],
                fig_pie_optimized=session_data.get('fig_pie_optimized'),
                fig_hist=session_data['fig_hist'],
                fig_dd=session_data['fig_dd'],
                fig_drift=session_data['fig_drift'],
                fig_dist=session_data['fig_dist']
            )
            
            # Create report data
            report_data = ReportFactory.create_simulation_report(
                tickers=session_data['all_tickers'],
                weights=session_data['weights'],
                simulation_results=session_data['results'],
                backtest_results=session_data['backtest_results'],
                charts=charts,
                horizon_years=session_data['horizon']
            )
            
            # Generate PDF
            pdf_generator = PDFReportGenerator()
            pdf_buffer = pdf_generator.generate(report_data)
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="portfolio_report.pdf",
                mime="application/pdf"
            )
    
    def _render_optimized_allocation(self, session_data: Dict[str, Any]):
        """Render optimized portfolio allocation section."""
        if session_data.get('fig_pie_optimized'):
            st.header('Optimized Portfolio Allocation')
            
            # Display optimized pie chart
            st.plotly_chart(session_data['fig_pie_optimized'], key="pie_chart_optimized_results")
            
            # Display optimized weights table
            optimized_weights = st.session_state.get('optimized_weights')
            all_tickers = st.session_state.get('all_tickers')
            
            if optimized_weights is not None and all_tickers is not None:
                # Create table of optimized weights
                weights_df = pd.DataFrame({
                    'Ticker': all_tickers,
                    'Optimized Weight': [f"{w:.4f}" for w in optimized_weights],
                    'Percentage': [f"{w*100:.2f}%" for w in optimized_weights]
                })
                
                st.subheader('Optimized Weights')
                st.dataframe(weights_df, use_container_width=True)
    
    def _render_csv_download_button(self, results: Dict[str, float]):
        """Render CSV download button."""
        csv_buffer = io.StringIO()
        pd.DataFrame([results]).to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download Simulation Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="portfolio_simulation_results.csv",
            mime="text/csv"
        ) 