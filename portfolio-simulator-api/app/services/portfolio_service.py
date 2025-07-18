"""
Portfolio management service.
"""

import io
import csv
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.models.portfolio import Portfolio
from app.schemas.portfolio import PortfolioCreate, PortfolioUpdate, PortfolioOptimizationResponse
from app.utils.financial_calcs import FinancialCalculator


class PortfolioService:
    """Service for portfolio management operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.calculator = FinancialCalculator()
    
    def create_portfolio(self, portfolio_data: PortfolioCreate, user_id: int) -> Portfolio:
        """Create a new portfolio."""
        portfolio = Portfolio(
            user_id=user_id,
            name=portfolio_data.name,
            description=portfolio_data.description,
            tickers=portfolio_data.tickers,
            weights=portfolio_data.weights,
            initial_investment=portfolio_data.initial_investment
        )
        
        self.db.add(portfolio)
        self.db.commit()
        self.db.refresh(portfolio)
        
        return portfolio
    
    def update_portfolio(self, portfolio: Portfolio, portfolio_data: PortfolioUpdate) -> Portfolio:
        """Update an existing portfolio."""
        update_data = portfolio_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(portfolio, field, value)
        
        self.db.commit()
        self.db.refresh(portfolio)
        
        return portfolio
    
    def get_user_portfolios(self, user_id: int) -> List[Portfolio]:
        """Get all portfolios for a user."""
        return self.db.query(Portfolio).filter(
            Portfolio.user_id == user_id
        ).order_by(Portfolio.created_at.desc()).all()
    
    def get_portfolio(self, portfolio_id: int, user_id: int) -> Optional[Portfolio]:
        """Get a specific portfolio for a user."""
        return self.db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id
        ).first()
    
    def delete_portfolio(self, portfolio_id: int, user_id: int) -> bool:
        """Delete a portfolio."""
        portfolio = self.get_portfolio(portfolio_id, user_id)
        if not portfolio:
            return False
        
        self.db.delete(portfolio)
        self.db.commit()
        return True
    
    async def optimize_weights(self, portfolio: Portfolio) -> PortfolioOptimizationResponse:
        """Optimize portfolio weights for maximum Sharpe ratio."""
        try:
            # Fetch historical data for optimization
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)  # 5 years of data
            
            data = await self.calculator.fetch_data(
                portfolio.tickers,
                start_date.strftime('%Y-%m-%d')
            )
            
            returns = self.calculator.calculate_returns(data, ter=0.001)
            optimized_weights = self.calculator.optimize_weights(returns)
            
            if optimized_weights is None:
                raise HTTPException(
                    status_code=400,
                    detail="Portfolio optimization failed. Please check your tickers."
                )
            
            # Calculate metrics for both portfolios
            original_metrics = self.calculator.calculate_portfolio_metrics(returns, portfolio.weights)
            optimized_metrics = self.calculator.calculate_portfolio_metrics(returns, optimized_weights)
            
            improvement = ((optimized_metrics['sharpe_ratio'] - original_metrics['sharpe_ratio']) 
                          / original_metrics['sharpe_ratio'] * 100)
            
            return PortfolioOptimizationResponse(
                portfolio_id=portfolio.id,
                original_weights=portfolio.weights,
                optimized_weights=optimized_weights.tolist(),
                expected_return=optimized_metrics['expected_return'],
                expected_volatility=optimized_metrics['expected_volatility'],
                sharpe_ratio=optimized_metrics['sharpe_ratio'],
                improvement_percentage=improvement
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {str(e)}"
            )
    
    async def process_csv_upload(self, file: UploadFile, user_id: int) -> Dict[str, Any]:
        """Process uploaded CSV file and extract portfolio data."""
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV file"
            )
        
        try:
            # Read CSV content
            content = await file.read()
            csv_string = content.decode('utf-8')
            
            # Detect CSV format (Degiro vs generic)
            if self._is_degiro_format(csv_string):
                return await self._process_degiro_csv(csv_string, user_id)
            else:
                return await self._process_generic_csv(csv_string, user_id)
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process CSV file: {str(e)}"
            )
    
    def _is_degiro_format(self, csv_content: str) -> bool:
        """Check if CSV is in Degiro transaction format."""
        first_line = csv_content.split('\n')[0].lower()
        degiro_headers = ['datum', 'tijd', 'product', 'isin', 'omschrijving']
        return any(header in first_line for header in degiro_headers)
    
    async def _process_degiro_csv(self, csv_content: str, user_id: int) -> Dict[str, Any]:
        """Process Degiro transaction CSV format."""
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Extract transactions (buy orders)
        buy_transactions = df[df['Omschrijving'].str.contains('Koop', na=False)]
        
        if buy_transactions.empty:
            raise HTTPException(
                status_code=400,
                detail="No buy transactions found in CSV"
            )
        
        # Group by ISIN and calculate totals
        portfolio_data = {}
        for _, transaction in buy_transactions.iterrows():
            isin = transaction.get('ISIN', '')
            ticker = transaction.get('Product', '')
            quantity = abs(float(transaction.get('Aantal', 0)))
            price = abs(float(transaction.get('Koers', 0)))
            
            if isin not in portfolio_data:
                portfolio_data[isin] = {
                    'ticker': ticker,
                    'total_quantity': 0,
                    'total_value': 0
                }
            
            portfolio_data[isin]['total_quantity'] += quantity
            portfolio_data[isin]['total_value'] += quantity * price
        
        # Convert to tickers and weights
        tickers = []
        values = []
        total_portfolio_value = sum(data['total_value'] for data in portfolio_data.values())
        
        for isin, data in portfolio_data.items():
            # Try to map ISIN to ticker if ticker not available
            ticker = data['ticker'] or self._map_isin_to_ticker(isin)
            if ticker:
                tickers.append(ticker)
                values.append(data['total_value'])
        
        if not tickers:
            raise HTTPException(
                status_code=400,
                detail="No valid tickers found in CSV"
            )
        
        weights = [value / total_portfolio_value for value in values]
        
        return {
            'tickers': tickers,
            'weights': weights,
            'initial_investment': total_portfolio_value,
            'detected_format': 'Degiro',
            'transactions_processed': len(buy_transactions)
        }
    
    async def _process_generic_csv(self, csv_content: str, user_id: int) -> Dict[str, Any]:
        """Process generic portfolio CSV format."""
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Expected columns: Ticker, Shares, Price (or Value)
        required_columns = ['ticker', 'shares']
        df.columns = df.columns.str.lower()
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_columns}"
            )
        
        tickers = df['ticker'].tolist()
        shares = df['shares'].tolist()
        
        # Calculate values
        if 'value' in df.columns:
            values = df['value'].tolist()
        elif 'price' in df.columns:
            values = [shares[i] * df['price'].iloc[i] for i in range(len(shares))]
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'value' or 'price' column"
            )
        
        total_value = sum(values)
        weights = [value / total_value for value in values]
        
        return {
            'tickers': tickers,
            'weights': weights,
            'initial_investment': total_value,
            'detected_format': 'Generic',
            'positions_processed': len(tickers)
        }
    
    def _map_isin_to_ticker(self, isin: str) -> Optional[str]:
        """Map ISIN to ticker symbol."""
        # ISIN to ticker mapping (extend this as needed)
        isin_mapping = {
            'IE00B4L5Y983': 'IWDA.AS',
            'IE00BHZRQZ17': 'FLXI.DE',
            'IE00B4ND3602': 'PPFB.DE',
            'IE00BZCQB185': 'QDV5.DE',
            'IE00B5BMR087': 'SXR8.DE',
            'IE00B3XXRP09': 'VUSA.AS',
            'US67066G1040': 'NVD.DE',
            'IE00BFY0GT14': 'SPPW.DE',
            'NL0010273215': 'ASML.AS',
            'IE000RHYOR04': 'ERNX.DE',
            'IE00B3FH7618': 'IEGE.MI',
            'LU0290358497': 'XEON.DE'
        }
        
        return isin_mapping.get(isin) 