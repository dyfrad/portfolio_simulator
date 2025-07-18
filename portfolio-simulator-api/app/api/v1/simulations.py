"""
Portfolio simulation endpoints.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.api.deps import get_current_user, get_current_premium_user
from app.models.user import User
from app.models.portfolio import Portfolio
from app.schemas.simulation import (
    SimulationRequest,
    SimulationResponse,
    SimulationStatus
)
from app.services.simulation_service import SimulationService

router = APIRouter()

@router.post("/run", response_model=SimulationResponse)
async def run_simulation(
    simulation_request: SimulationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Run Monte Carlo portfolio simulation."""
    # Get portfolio
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == simulation_request.portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Check simulation limits based on user type
    max_simulations = (
        settings.MAX_SIMULATIONS if current_user.is_premium 
        else settings.DEFAULT_SIMULATIONS
    )
    
    if simulation_request.num_simulations > max_simulations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {max_simulations} simulations allowed for your plan"
        )
    
    # Initialize simulation service
    service = SimulationService(db)
    
    # Run simulation
    try:
        result = await service.run_monte_carlo_simulation(
            portfolio=portfolio,
            config=simulation_request,
            user_id=current_user.id
        )
        
        # Store result in background
        background_tasks.add_task(
            service.save_simulation_result,
            result,
            current_user.id,
            portfolio.id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )

@router.get("/")
def get_user_simulations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get user's simulation history."""
    service = SimulationService(db)
    return service.get_user_simulation_history(current_user.id)

@router.get("/history")
def get_simulation_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get user's simulation history."""
    service = SimulationService(db)
    return service.get_user_simulation_history(current_user.id)

@router.get("/{simulation_id}")
def get_simulation_result(
    simulation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get specific simulation result."""
    service = SimulationService(db)
    result = service.get_simulation_result(simulation_id, current_user.id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation result not found"
        )
    
    return result

@router.get("/progress/{simulation_id}")
def get_simulation_progress(
    simulation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get simulation progress."""
    service = SimulationService(db)
    progress = service.get_simulation_progress(simulation_id, current_user.id)
    
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found"
        )
    
    return progress

@router.post("/stress-test")
async def run_stress_test(
    simulation_request: SimulationRequest,
    current_user: User = Depends(get_current_premium_user),  # Premium only
    db: Session = Depends(get_db)
) -> Any:
    """Run stress test simulation (premium feature)."""
    # Get portfolio
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == simulation_request.portfolio_id,
        Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    service = SimulationService(db)
    
    try:
        result = await service.run_stress_test_simulation(
            portfolio=portfolio,
            config=simulation_request,
            user_id=current_user.id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress test failed: {str(e)}"
        ) 