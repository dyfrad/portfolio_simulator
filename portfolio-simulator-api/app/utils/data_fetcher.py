"""
Data fetching utilities for external financial data sources.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from app.core.config import settings


class DataFetcher:
    """Handles fetching financial data from external sources."""
    
    def __init__(self):
        self.base_url = settings.YAHOO_FINANCE_BASE_URL
        self.timeout = settings.YAHOO_FINANCE_TIMEOUT
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_yahoo_finance_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data from Yahoo Finance API.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        session = await self._get_session()
        
        # Fetch data for all tickers concurrently
        tasks = []
        for ticker in tickers:
            task = self._fetch_single_ticker(session, ticker, start_timestamp, end_timestamp)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                print(f"Failed to fetch data for {ticker}: {str(result)}")
                continue
            
            if result is not None:
                data[ticker] = result
        
        if not data:
            raise ValueError("Failed to fetch data for any tickers")
        
        return data
    
    async def _fetch_single_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        start_timestamp: int,
        end_timestamp: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker."""
        url = f"{self.base_url}/v8/finance/chart/{ticker}"
        
        params = {
            'period1': start_timestamp,
            'period2': end_timestamp,
            'interval': '1d',
            'includePrePost': 'false',
            'events': 'div,splits'
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                data = await response.json()
                return self._parse_yahoo_response(data, ticker)
                
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            return None
    
    def _parse_yahoo_response(self, data: Dict[str, Any], ticker: str) -> Optional[pd.DataFrame]:
        """Parse Yahoo Finance API response."""
        try:
            chart = data['chart']['result'][0]
            
            # Extract timestamps and indicators
            timestamps = chart['timestamp']
            indicators = chart['indicators']['quote'][0]
            
            # Get OHLCV data
            opens = indicators.get('open', [])
            highs = indicators.get('high', [])
            lows = indicators.get('low', [])
            closes = indicators.get('close', [])
            volumes = indicators.get('volume', [])
            
            # Handle adjusted close if available
            adj_close = closes
            if 'adjclose' in chart['indicators'] and chart['indicators']['adjclose']:
                adj_close = chart['indicators']['adjclose'][0]['adjclose']
            
            # Create DataFrame
            df_data = []
            for i, timestamp in enumerate(timestamps):
                if i < len(closes) and closes[i] is not None:
                    df_data.append({
                        'date': pd.to_datetime(timestamp, unit='s'),
                        'open': opens[i] if i < len(opens) and opens[i] is not None else np.nan,
                        'high': highs[i] if i < len(highs) and highs[i] is not None else np.nan,
                        'low': lows[i] if i < len(lows) and lows[i] is not None else np.nan,
                        'close': closes[i],
                        'adj_close': adj_close[i] if i < len(adj_close) and adj_close[i] is not None else closes[i],
                        'volume': volumes[i] if i < len(volumes) and volumes[i] is not None else 0
                    })
            
            if not df_data:
                return None
            
            df = pd.DataFrame(df_data)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Use adjusted close as the main close price
            df['close'] = df['adj_close']
            
            # Remove rows with missing close prices
            df = df.dropna(subset=['close'])
            
            if len(df) < 10:  # Minimum data requirement
                return None
            
            return df
            
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing data for {ticker}: {str(e)}")
            return None
    
    async def fetch_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current market prices for tickers."""
        session = await self._get_session()
        
        # Use quote endpoint for current prices
        url = f"{self.base_url}/v7/finance/quote"
        params = {'symbols': ','.join(tickers)}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                data = await response.json()
                quotes = data['quoteResponse']['result']
                
                prices = {}
                for quote in quotes:
                    symbol = quote['symbol']
                    # Try different price fields
                    price = (quote.get('regularMarketPrice') or 
                            quote.get('previousClose') or 
                            quote.get('bid') or 
                            quote.get('ask'))
                    
                    if price is not None:
                        prices[symbol] = float(price)
                
                return prices
                
        except Exception as e:
            print(f"Error fetching current prices: {str(e)}")
            return {}
    
    async def validate_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """Validate if tickers exist and have data."""
        session = await self._get_session()
        
        url = f"{self.base_url}/v7/finance/quote"
        params = {'symbols': ','.join(tickers)}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {ticker: False for ticker in tickers}
                
                data = await response.json()
                quotes = data['quoteResponse']['result']
                
                valid_tickers = {quote['symbol']: True for quote in quotes}
                
                # Mark missing tickers as invalid
                result = {}
                for ticker in tickers:
                    result[ticker] = valid_tickers.get(ticker, False)
                
                return result
                
        except Exception as e:
            print(f"Error validating tickers: {str(e)}")
            return {ticker: False for ticker in tickers}
    
    async def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a ticker."""
        session = await self._get_session()
        
        url = f"{self.base_url}/v7/finance/quote"
        params = {'symbols': ticker}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                quotes = data['quoteResponse']['result']
                
                if not quotes:
                    return None
                
                quote = quotes[0]
                
                return {
                    'symbol': quote.get('symbol'),
                    'longName': quote.get('longName'),
                    'shortName': quote.get('shortName'),
                    'currency': quote.get('currency'),
                    'exchange': quote.get('fullExchangeName'),
                    'marketCap': quote.get('marketCap'),
                    'trailingPE': quote.get('trailingPE'),
                    'dividendYield': quote.get('dividendYield'),
                    'sector': quote.get('sector'),
                    'industry': quote.get('industry')
                }
                
        except Exception as e:
            print(f"Error getting ticker info for {ticker}: {str(e)}")
            return None 