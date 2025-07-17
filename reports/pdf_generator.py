"""
PDF Report Generator for portfolio simulation reports.
"""

import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .data_models import ReportData


class PDFReportGenerator:
    """Generates PDF reports for portfolio simulation results."""
    
    def __init__(self, page_size=letter):
        """Initialize with configurable page size."""
        self.page_size = page_size
        self.width, self.height = page_size
        self.margin = 100
        self.line_height = 15
        
    def generate(self, report_data: ReportData) -> io.BytesIO:
        """Generate complete PDF report from structured data."""
        buffer = io.BytesIO()
        c = self._create_canvas(buffer)
        
        self._add_title_page(c, report_data)
        self._add_allocation_section(c, report_data)
        self._add_results_sections(c, report_data)
        self._add_charts_sections(c, report_data)
        
        c.save()
        buffer.seek(0)
        return buffer
    
    def _create_canvas(self, buffer: io.BytesIO) -> canvas.Canvas:
        """Create PDF canvas with consistent settings."""
        return canvas.Canvas(buffer, pagesize=self.page_size)
    
    def _add_title_page(self, c: canvas.Canvas, report_data: ReportData) -> None:
        """Add title and header information."""
        y = self.height - 50
        c.drawString(self.margin, y, "Portfolio Simulator Report")
        y -= 30
        c.drawString(self.margin, y, f"Time Horizon: {report_data.horizon_years} years")
    
    def _add_allocation_section(self, c: canvas.Canvas, report_data: ReportData) -> None:
        """Add portfolio allocation chart and details."""
        y = self.height - 120
        c.drawString(self.margin, y, "Portfolio Allocation:")
        y -= 20
        
        # Add pie chart
        if 'pie' in report_data.charts:
            pie_img = self._chart_to_image(report_data.charts['pie'])
            if pie_img:
                c.drawImage(pie_img, self.margin, y - 200, width=400, height=200)
    
    def _add_results_sections(self, c: canvas.Canvas, report_data: ReportData) -> None:
        """Add simulation and backtest results."""
        y = self.height - 370
        
        # Simulation results
        c.drawString(self.margin, y, "Simulation Results:")
        y -= 20
        for key in report_data.simulation_results:
            formatted_value = report_data.format_simulation_result(key)
            c.drawString(self.margin + 20, y, f"{key}: {formatted_value}")
            y -= self.line_height
        
        y -= 20
        
        # Backtest results  
        c.drawString(self.margin, y, "Backtesting Results:")
        y -= 20
        for key in report_data.backtest_results:
            formatted_value = report_data.format_backtest_result(key)
            c.drawString(self.margin + 20, y, f"{key}: {formatted_value}")
            y -= self.line_height
    
    def _add_charts_sections(self, c: canvas.Canvas, report_data: ReportData) -> None:
        """Add charts on separate pages."""
        chart_names = ['hist', 'dd', 'drift', 'dist']
        chart_titles = [
            'Historical Performance', 
            'Drawdown Analysis',
            'Weight Drift Analysis', 
            'Distribution of Outcomes'
        ]
        
        for chart_name, title in zip(chart_names, chart_titles):
            if chart_name in report_data.charts:
                c.showPage()
                y = self.height - 50
                c.drawString(self.margin, y, title)
                y -= 30
                
                chart_img = self._chart_to_image(report_data.charts[chart_name])
                if chart_img:
                    c.drawImage(chart_img, self.margin, y - 350, width=400, height=300)
    
    def _chart_to_image(self, chart) -> ImageReader:
        """Convert chart object to ImageReader for PDF insertion."""
        try:
            img_buffer = io.BytesIO()
            
            # Handle different chart types
            if hasattr(chart, 'write_image'):  # Plotly chart
                chart.write_image(img_buffer, format='png')
            elif hasattr(chart, 'savefig'):  # Matplotlib chart
                chart.savefig(img_buffer, format='png')
            else:
                return None
                
            img_buffer.seek(0)
            return ImageReader(img_buffer)
        except Exception:
            return None 