"""
Report generation endpoints.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.report import ReportRequest, ReportResponse
from app.services.report_service import ReportService

router = APIRouter()

@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Generate PDF report for simulation results."""
    service = ReportService(db)
    
    try:
        # Start report generation in background
        report_id = await service.generate_report_async(
            report_request,
            current_user.id,
            background_tasks
        )
        
        return {
            "report_id": report_id,
            "status": "generating",
            "message": "Report generation started"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )

@router.get("/{report_id}/status")
def get_report_status(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get report generation status."""
    service = ReportService(db)
    status_info = service.get_report_status(report_id, current_user.id)
    
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    return status_info

@router.get("/{report_id}/download")
def download_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FileResponse:
    """Download generated PDF report."""
    service = ReportService(db)
    file_path = service.get_report_file_path(report_id, current_user.id)
    
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found or not ready"
        )
    
    return FileResponse(
        path=file_path,
        filename=f"portfolio_report_{report_id}.pdf",
        media_type="application/pdf"
    )

@router.get("/")
def get_user_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get all reports for current user."""
    service = ReportService(db)
    return service.get_user_reports(current_user.id) 