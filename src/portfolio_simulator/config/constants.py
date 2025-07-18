"""
Constants for portfolio simulator.
"""

# Default tickers for the assets (UCITS-compliant, EUR-traded)
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']  # MSCI World, MSCI India, Gold, Cash

# ISIN to Ticker mapping
ISIN_TO_TICKER = {
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

# Default start date for historical data
DEFAULT_START_DATE = '2015-01-01'