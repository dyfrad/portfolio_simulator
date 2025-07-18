"""
Report generation service for PDF and Excel reports.
"""

import uuid
import os
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
from app.models.simulation import SimulationResult
from app.schemas.report import ReportRequest, ReportType, ReportFormat
from app.core.config import settings


class ReportService:
    """Service for generating and managing reports."""
    
    def __init__(self, db: Session):
        self.db = db
        self.reports_storage_path = settings.REPORTS_STORAGE_PATH
        
        # Ensure reports directory exists
        os.makedirs(self.reports_storage_path, exist_ok=True)
    
    async def generate_report_async(
        self,
        report_request: ReportRequest,
        user_id: int,
        background_tasks: BackgroundTasks
    ) -> str:
        """Start report generation in background."""
        report_id = str(uuid.uuid4())
        
        # Add background task for report generation
        background_tasks.add_task(
            self._generate_report_task,
            report_id,
            report_request,
            user_id
        )
        
        return report_id
    
    async def _generate_report_task(
        self,
        report_id: str,
        report_request: ReportRequest,
        user_id: int
    ) -> None:
        """Background task for report generation."""
        try:
            # Get simulation result
            simulation_result = self.db.query(SimulationResult).filter(
                SimulationResult.simulation_id == report_request.simulation_id,
                SimulationResult.user_id == user_id
            ).first()
            
            if not simulation_result:
                raise Exception("Simulation result not found")
            
            if report_request.report_format == ReportFormat.PDF:
                await self._generate_pdf_report(
                    report_id,
                    simulation_result,
                    report_request
                )
            elif report_request.report_format == ReportFormat.EXCEL:
                await self._generate_excel_report(
                    report_id,
                    simulation_result,
                    report_request
                )
            
        except Exception as e:
            # Log error and mark report as failed
            print(f"Report generation failed: {str(e)}")
    
    async def _generate_pdf_report(
        self,
        report_id: str,
        simulation_result: SimulationResult,
        report_request: ReportRequest
    ) -> None:
        """Generate PDF report."""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        import json
        
        # File path
        filename = f"portfolio_report_{report_id}.pdf"
        file_path = os.path.join(self.reports_storage_path, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = report_request.custom_title or "Portfolio Simulation Report"
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Simulation metadata
        config = simulation_result.config
        results = simulation_result.results
        
        # Portfolio information
        story.append(Paragraph("Portfolio Information", styles['Heading2']))
        portfolio_data = [
            ['Parameter', 'Value'],
            ['Portfolio ID', str(simulation_result.portfolio_id)],
            ['Simulation Date', simulation_result.created_at.strftime('%Y-%m-%d %H:%M')],
            ['Time Horizon', f"{config.get('horizon_years', 0)} years"],
            ['Number of Simulations', f"{config.get('num_simulations', 0):,}"],
            ['Inflation Rate', f"{config.get('inflation_rate', 0):.2%}"],
        ]
        
        portfolio_table = Table(portfolio_data)
        portfolio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(portfolio_table)
        story.append(Spacer(1, 12))
        
        # Results summary
        story.append(Paragraph("Simulation Results", styles['Heading2']))
        results_data = [
            ['Metric', 'Value'],
            ['Mean Final Value', f"€{results.get('mean_final_value', 0):,.2f}"],
            ['Median Final Value', f"€{results.get('median_final_value', 0):,.2f}"],
            ['Standard Deviation', f"€{results.get('std_final_value', 0):,.2f}"],
            ['Value at Risk (95%)', f"€{results.get('var_95', 0):,.2f}"],
            ['Conditional VaR (95%)', f"€{results.get('cvar_95', 0):,.2f}"],
            ['Mean Total Return', f"{results.get('mean_total_return', 0):.2%}"],
            ['Probability of Loss', f"{results.get('probability_of_loss', 0):.2%}"],
        ]
        
        results_table = Table(results_data)
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
        story.append(Spacer(1, 12))
        
        # Risk analysis
        story.append(Paragraph("Risk Analysis", styles['Heading2']))
        min_value = results.get('min_final_value', 0)
        max_value = results.get('max_final_value', 0)
        
        risk_text = f"""
        Based on {config.get('num_simulations', 0):,} Monte Carlo simulations over {config.get('horizon_years', 0)} years:
        
        • Best case scenario: €{max_value:,.2f}
        • Worst case scenario: €{min_value:,.2f}
        • 95% confidence interval: €{results.get('var_95', 0):,.2f} - €{max_value:,.2f}
        
        The portfolio shows a {results.get('probability_of_loss', 0):.1%} probability of loss over the investment horizon.
        """
        
        story.append(Paragraph(risk_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
    
    async def _generate_excel_report(
        self,
        report_id: str,
        simulation_result: SimulationResult,
        report_request: ReportRequest
    ) -> None:
        """Generate Excel report."""
        import pandas as pd
        
        filename = f"portfolio_report_{report_id}.xlsx"
        file_path = os.path.join(self.reports_storage_path, filename)
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            config = simulation_result.config
            results = simulation_result.results
            
            summary_data = {
                'Metric': [
                    'Portfolio ID',
                    'Simulation Date',
                    'Time Horizon (years)',
                    'Number of Simulations',
                    'Mean Final Value',
                    'Median Final Value',
                    'Standard Deviation',
                    'Value at Risk (95%)',
                    'Conditional VaR (95%)',
                    'Mean Total Return',
                    'Probability of Loss'
                ],
                'Value': [
                    simulation_result.portfolio_id,
                    simulation_result.created_at.strftime('%Y-%m-%d %H:%M'),
                    config.get('horizon_years', 0),
                    config.get('num_simulations', 0),
                    results.get('mean_final_value', 0),
                    results.get('median_final_value', 0),
                    results.get('std_final_value', 0),
                    results.get('var_95', 0),
                    results.get('cvar_95', 0),
                    results.get('mean_total_return', 0),
                    results.get('probability_of_loss', 0)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Raw data sheet (if requested)
            if report_request.include_raw_data:
                final_values = results.get('final_values', [])
                if final_values:
                    raw_data_df = pd.DataFrame({'Final Values': final_values})
                    raw_data_df.to_excel(writer, sheet_name='Raw Data', index=False)
    
    def get_report_status(self, report_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Get report generation status."""
        # Check if file exists
        pdf_path = os.path.join(self.reports_storage_path, f"portfolio_report_{report_id}.pdf")
        excel_path = os.path.join(self.reports_storage_path, f"portfolio_report_{report_id}.xlsx")
        
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            return {
                'report_id': report_id,
                'status': 'completed',
                'file_size_bytes': file_size,
                'download_url': f"/api/v1/reports/{report_id}/download"
            }
        elif os.path.exists(excel_path):
            file_size = os.path.getsize(excel_path)
            return {
                'report_id': report_id,
                'status': 'completed',
                'file_size_bytes': file_size,
                'download_url': f"/api/v1/reports/{report_id}/download"
            }
        else:
            return {
                'report_id': report_id,
                'status': 'generating',
                'progress_percentage': 50.0
            }
    
    def get_report_file_path(self, report_id: str, user_id: int) -> Optional[str]:
        """Get file path for report download."""
        pdf_path = os.path.join(self.reports_storage_path, f"portfolio_report_{report_id}.pdf")
        excel_path = os.path.join(self.reports_storage_path, f"portfolio_report_{report_id}.xlsx")
        
        if os.path.exists(pdf_path):
            return pdf_path
        elif os.path.exists(excel_path):
            return excel_path
        
        return None
    
    def get_user_reports(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all reports for a user."""
        # This would typically query a reports table
        # For now, return empty list as we're not storing report metadata
        return [] 