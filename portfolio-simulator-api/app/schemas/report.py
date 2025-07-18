"""
Report generation schemas.
"""

from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ReportType(str, Enum):
    """Available report types."""
    SIMULATION_SUMMARY = "simulation_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    PORTFOLIO_COMPARISON = "portfolio_comparison"


class ReportFormat(str, Enum):
    """Available report formats."""
    PDF = "pdf"
    EXCEL = "excel"


class ReportRequest(BaseModel):
    """Schema for report generation request."""
    simulation_id: str
    report_type: ReportType = ReportType.SIMULATION_SUMMARY
    report_format: ReportFormat = ReportFormat.PDF
    
    # Optional customization
    include_charts: bool = True
    include_raw_data: bool = False
    custom_title: Optional[str] = None
    
    @validator('custom_title')
    def validate_custom_title(cls, v):
        if v is not None and len(v) > 100:
            raise ValueError('Custom title cannot exceed 100 characters')
        return v


class ReportResponse(BaseModel):
    """Schema for report generation response."""
    report_id: str
    status: str
    message: str
    download_url: Optional[str] = None
    created_at: datetime


class ReportStatusResponse(BaseModel):
    """Schema for report status response."""
    report_id: str
    status: str  # "generating", "completed", "failed"
    progress_percentage: Optional[float] = None
    message: Optional[str] = None
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class UserReportResponse(BaseModel):
    """Schema for user reports list response."""
    report_id: str
    simulation_id: str
    portfolio_name: str
    report_type: ReportType
    report_format: ReportFormat
    status: str
    file_size_bytes: Optional[int] = None
    created_at: datetime
    download_url: Optional[str] = None 