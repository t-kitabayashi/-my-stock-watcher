import streamlit as st
import yfinance as yf
import pandas as pd

# Page config
st.set_page_config(page_title="Stock Watcher", layout="wide")

st.title("📊 Stock Watcher")

# Ticker definitions with User-friendly names and base currency
TICKERS = {
    "^GSPC": {"name": "S&P 500 (eMAXIS Slim/SBI・V)", "currency": "USD"},
    "VT": {"name": "全世界株式 (All Country)", "currency": "USD"},
    "VEA": {"name": "先進国株式 (Developed Markets)", "currency": "USD"},
    "1489.T": {"name": "日本高配当 (NF・日経高配当50)", "currency": "JPY"},
    "GC=F": {"name": "金 (Gold)", "currency": "USD"},
    "VYM": {"name": "米国高配当 (High Div Yield)", "currency": "USD"},
    "VTI": {"name": "全米株式 (Total Stock Mkt)", "currency": "USD"},
    "^NYFANG": {"name": "FANG+ (iFreeNEXT FANG+)", "currency": "USD"}
}

def get_data():
    """Fetches data for all tickers plus USD/JPY."""
    ticker_list = list(TICKERS.keys()) + ["USDJPY=X"]
    # Fetch 15 months to get daily, monthly, and yearly changes
    data = yf.download(ticker_list, period="15mo", group_by='ticker', threads=True)
    return data

# Initialize session state for the selected ticker
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = list(TICKERS.keys())[0]

# Sidebar for Settings
st.sidebar.title("Settings")
view_mode = st.sidebar.radio("表示モード (View Mode)", ["カード形式", "リスト形式"], index=0)
display_currency = st.sidebar.radio("表示通貨 (Display Currency)", ["JPY", "USD"], index=0)

if st.button('🔄 Update Data'):
    st.cache_data.clear()

# Fetch data
with st.spinner('Fetching latest stock data...'):
    try:
        df = get_data()
        usdjpy_rate = df["USDJPY=X"]["Close"].dropna().iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

st.sidebar.info(f"現在の為替: 1 USD = {usdjpy_rate:.2f} JPY")

# Conversion Helper
def convert_value(val, from_curr, to_curr, rate):
    if from_curr == to_curr:
        return val
    if from_curr == "USD" and to_curr == "JPY":
        return val * rate
    if from_curr == "JPY" and to_curr == "USD":
        return val / rate
    return val

# Display logic
st.subheader(f"Market Overview ({display_currency})")

if view_mode == "カード形式":
    # Create a grid layout
    cols = st.columns(4) # 4 columns per row

    for index, (symbol, info) in enumerate(TICKERS.items()):
        col = cols[index % 4]
        name = info["name"]
        base_curr = info["currency"]
        
        with col:
            ticker_data = df[symbol]
            valid_data = ticker_data['Close'].dropna()
            
            if len(valid_data) >= 2:
                raw_current = valid_data.iloc[-1]
                raw_prev = valid_data.iloc[-2]
                
                # Convert values
                current_price = convert_value(raw_current, base_curr, display_currency, usdjpy_rate)
                prev_price = convert_value(raw_prev, base_curr, display_currency, usdjpy_rate)
                
                delta = current_price - prev_price
                delta_percent = (delta / prev_price) * 100
                
                symbol_mark = "¥" if display_currency == "JPY" else "$"
                st.metric(
                    label=name,
                    value=f"{symbol_mark}{current_price:,.2f}",
                    delta=f"{delta:,.2f} ({delta_percent:.2f}%)"
                )
            else:
                st.metric(label=name, value="N/A", delta="No Data")
            
            # Selection button for the chart
            button_label = name.split(' ')[0]
            if st.button(f"📊 {button_label} グラフ", key=f"btn_card_{symbol}"):
                st.session_state.selected_ticker = symbol
else:
    # List View (Table Mode)
    summary_data = []
    symbol_mark = "¥" if display_currency == "JPY" else "$"
    
    for symbol, info in TICKERS.items():
        name = info["name"]
        base_curr = info["currency"]
        ticker_data = df[symbol]
        valid_data = ticker_data['Close'].dropna()
        
        if len(valid_data) >= 2:
            raw_current = valid_data.iloc[-1]
            raw_prev_day = valid_data.iloc[-2]
            
            # Find price 1 month ago (approx 20 trading days ago or use time based)
            # We fetch 2mo, so we can look back ~30 days
            latest_date = valid_data.index[-1]
            month_ago_date = latest_date - pd.Timedelta(days=30)
            # Find nearest date that is <= month_ago_date
            month_ago_data = valid_data[valid_data.index <= month_ago_date]
            if not month_ago_data.empty:
                raw_prev_month = month_ago_data.iloc[-1]
            else:
                raw_prev_month = valid_data.iloc[0] # Earliest available
            
            # Find price 1 year ago
            year_ago_date = latest_date - pd.Timedelta(days=365)
            year_ago_data = valid_data[valid_data.index <= year_ago_date]
            if not year_ago_data.empty:
                raw_prev_year = year_ago_data.iloc[-1]
            else:
                raw_prev_year = valid_data.iloc[0] # Earliest available
            
            curr_val = convert_value(raw_current, base_curr, display_currency, usdjpy_rate)
            day_val = convert_value(raw_prev_day, base_curr, display_currency, usdjpy_rate)
            month_val = convert_value(raw_prev_month, base_curr, display_currency, usdjpy_rate)
            year_val = convert_value(raw_prev_year, base_curr, display_currency, usdjpy_rate)
            
            day_chg = ((curr_val - day_val) / day_val) * 100
            month_chg = ((curr_val - month_val) / month_val) * 100
            year_chg = ((curr_val - year_val) / year_val) * 100
            
            summary_data.append({
                "銘柄名": name,
                f"現在値 ({display_currency})": f"{symbol_mark}{curr_val:,.2f}",
                "前日比 (%)": day_chg,
                "前月比 (%)": month_chg,
                "前年比 (%)": year_chg,
                "Symbol": symbol
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Custom display with st.dataframe and styling
    def color_negative_red(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'

    st.dataframe(
        summary_df.drop(columns=["Symbol"]).style.format({
            "前日比 (%)": "{:+.2f}%",
            "前月比 (%)": "{:+.2f}%",
            "前年比 (%)": "{:+.2f}%"
        }).applymap(color_negative_red, subset=["前日比 (%)", "前月比 (%)", "前年比 (%)"]),
        use_container_width=True,
        hide_index=True
    )
    
    # Selection radio for List View
    st.write("詳細を表示する銘柄を選択してください:")
    selected_list_name = st.radio(
        "Ticker Selection", 
        options=summary_df["銘柄名"].tolist(),
        horizontal=True,
        label_visibility="collapsed"
    )
    # Update selected_ticker session state based on radio selection
    st.session_state.selected_ticker = summary_df[summary_df["銘柄名"] == selected_list_name]["Symbol"].values[0]

# Historical Chart Section
st.divider()
selected_symbol = st.session_state.selected_ticker
selected_name = TICKERS[selected_symbol]["name"]
selected_base_curr = TICKERS[selected_symbol]["currency"]

st.subheader(f"📈 {selected_name} - Historical Chart ({display_currency})")

# Options for Chart
analysis_period = st.selectbox(
    "分析・予測期間を選択",
    ["なし (表示しない)", "3ヶ月", "6ヶ月", "1年", "3年", "5年"],
    index=0
)
show_analysis = analysis_period != "なし (表示しない)"

# Fetch historical data (1 year)
@st.cache_data(ttl=3600)
def get_historical_with_fx(ticker):
    # Fetch data up to 5 years
    t_obj = yf.Ticker(ticker)
    fx_obj = yf.Ticker("USDJPY=X")
    
    t_data = t_obj.history(period="5y")
    fx_data = fx_obj.history(period="5y")
    
    if t_data.empty:
        return pd.DataFrame()

    # Create a unified date index (normalize time part)
    t_data.index = pd.to_datetime(t_data.index).tz_localize(None).normalize()
    fx_data.index = pd.to_datetime(fx_data.index).tz_localize(None).normalize()
    
    # Use reindex to align FX data to the stock data's trading days, 
    # then forward fill any gaps in FX (like holidays that differ)
    fx_aligned = fx_data["Close"].reindex(t_data.index, method='ffill')
    
    combined = pd.DataFrame({
        "Price": t_data["Close"],
        "FX": fx_aligned
    })
    
    # If the last FX is still NaN (unlikely but possible), backfill or use latest rate
    combined["FX"] = combined["FX"].ffill().bfill()
    
    return combined

hist_df = get_historical_with_fx(selected_symbol)

if not hist_df.empty:
    # Use copy to avoid modifying cache
    plot_df = hist_df.copy()
    
    # Conversion for historical
    if selected_base_curr == "USD" and display_currency == "JPY":
        plot_df["Display"] = plot_df["Price"] * plot_df["FX"]
    elif selected_base_curr == "JPY" and display_currency == "USD":
        plot_df["Display"] = plot_df["Price"] / plot_df["FX"]
    else:
        plot_df["Display"] = plot_df["Price"]

    # Calculate Moving Averages
    plot_df["3m MA"] = plot_df["Display"].rolling(window=60).mean()
    plot_df["6m MA"] = plot_df["Display"].rolling(window=120).mean()
    plot_df["1y MA"] = plot_df["Display"].rolling(window=240).mean()
    plot_df["3y MA"] = plot_df["Display"].rolling(window=240*3).mean()
    plot_df["5y MA"] = plot_df["Display"].rolling(window=240*5).mean()

    # Prediction Logic
    if show_analysis:
        import numpy as np
        # Define period mapping
        period_map = {
            "3ヶ月": {"days": 90, "ma_col": "3m MA"},
            "6ヶ月": {"days": 180, "ma_col": "6m MA"},
            "1年": {"days": 365, "ma_col": "1y MA"},
            "3年": {"days": 365*3, "ma_col": "3y MA"},
            "5年": {"days": 365*5, "ma_col": "5y MA"}
        }
        p_info = period_map[analysis_period]
        forecast_days = p_info["days"]
        ma_col = p_info["ma_col"]

        # Linear Regression on available data
        y_data = plot_df["Display"].dropna()
        x_data = np.arange(len(y_data))
        
        # Fit a line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        
        # Future projection
        last_date_val = plot_df.index[-1]
        future_dates = [last_date_val + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_x = np.arange(len(y_data), len(y_data) + forecast_days)
        future_preds = p(future_x)
        
        # Predicted value & Percentage change
        current_price_val = y_data.iloc[-1]
        pred_val = future_preds[-1]
        pct_change = ((pred_val - current_price_val) / current_price_val) * 100
        symbol_mark = "¥" if display_currency == "JPY" else "$"
        st.success(f"📊 {analysis_period}後の予測値 ({display_currency}): {symbol_mark}{pred_val:,.2f} (現在比: {pct_change:+.2f}%)")
        
        # Add prediction to DataFrame for plotting
        pred_df = pd.DataFrame({
            "Display": [np.nan] * forecast_days,
            "3m MA": [np.nan] * forecast_days,
            "6m MA": [np.nan] * forecast_days,
            "1y MA": [np.nan] * forecast_days,
            "3y MA": [np.nan] * forecast_days,
            "5y MA": [np.nan] * forecast_days,
            "Projection": future_preds
        }, index=future_dates)
        
        # Merge with main df for plotting
        plot_df["Projection"] = p(x_data) # Historical trend line
        plot_df = pd.concat([plot_df, pred_df])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["3ヶ月", "6ヶ月", "1年", "3年", "5年"])
    last_date_actual = hist_df.index[-1]
    
    # Prepare columns to show
    cols_to_plot = ["Display"]
    if show_analysis:
        cols_to_plot.append(ma_col)
        cols_to_plot.append("Projection")

    with tab1:
        mask = plot_df.index > (last_date_actual - pd.Timedelta(days=90))
        st.line_chart(plot_df.loc[mask, cols_to_plot])
        
    with tab2:
        mask = plot_df.index > (last_date_actual - pd.Timedelta(days=180))
        st.line_chart(plot_df.loc[mask, cols_to_plot])
        
    with tab3:
        mask = plot_df.index > (last_date_actual - pd.Timedelta(days=365))
        st.line_chart(plot_df.loc[mask, cols_to_plot])

    with tab4:
        mask = plot_df.index > (last_date_actual - pd.Timedelta(days=365*3))
        st.line_chart(plot_df.loc[mask, cols_to_plot])

    with tab5:
        # Full 5 years
        st.line_chart(plot_df[cols_to_plot])
else:
    st.warning("No historical data available for this ticker.")

st.caption(f"Data provided by yfinance. FX rate used for conversion: {usdjpy_rate:.2f}")
