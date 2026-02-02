import yfinance as yf

# Ticker definitions
TICKERS = {
    "^GSPC": "S&P 500",
    "VT": "ACWI",
    "VEA": "Developed",
    "1489.T": "Japan High Dividend",
    "GC=F": "Gold",
    "VYM": "US High Dividend",
    "VTI": "US Total Stock",
    "^NYFANG": "FANG+"
}

def verify_data():
    print("Fetching data...")
    ticker_list = list(TICKERS.keys())
    # Fetch data
    try:
        data = yf.download(ticker_list, period="5d", group_by='ticker', threads=True)
        if data.empty:
            print("Data fetch returned empty DataFrame.")
            return

        print("Data fetched successfully. Checking columns...")
        for ticker in ticker_list:
            if ticker in data.columns:
                df = data[ticker]
                last_price = df['Close'].dropna().iloc[-1] if not df['Close'].dropna().empty else "N/A"
                print(f"{ticker}: {last_price}")
            else:
                print(f"{ticker}: No data column found (might be fetching failure or different column structure)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_data()
