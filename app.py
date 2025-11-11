# app_present_focused.py - Present-focused analysis with historical support
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NSE Swing Trading System | Real-Time Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }

    .present-indicator {
        background-color: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    .historical-indicator {
        background-color: #6c757d;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }

    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stock-card-buy {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #4caf50;
    }

    .stock-card-sell {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #f44336;
    }

    .stock-card-hold {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #9e9e9e;
    }
</style>
""", unsafe_allow_html=True)

# Import your pipeline classes
try:
    from main_pipeline import SwingTradingSystem, NSEDataProcessor, FeatureEngineer
except ImportError:
    st.error("Cannot import required modules. Please check main_pipeline.py")
    st.stop()


# Enhanced Data Processor with proper date handling
class EnhancedDataProcessor(NSEDataProcessor):
    def __init__(self, base_path="./data"):
        super().__init__(base_path)
        self.all_dates = self.get_all_sorted_dates()
        self.most_recent_date = self.all_dates[-1] if self.all_dates else None

    def get_all_sorted_dates(self):
        """Get all dates sorted properly (oldest to newest)"""
        date_folders = []
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir() and len(item.name) == 8 and item.name.isdigit():
                    date_folders.append(item.name)

        # Sort by converting to YYYYMMDD format
        def sort_key(date_str):
            return date_str[4:8] + date_str[2:4] + date_str[0:2]

        return sorted(date_folders, key=sort_key)

    def format_date(self, date_str):
        """Convert DDMMYYYY to readable format"""
        try:
            day = date_str[0:2]
            month = date_str[2:4]
            year = date_str[4:8]
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime("%d-%b-%Y")
        except:
            return date_str

    def get_date_object(self, date_str):
        """Convert DDMMYYYY to datetime object"""
        try:
            day = int(date_str[0:2])
            month = int(date_str[2:4])
            year = int(date_str[4:8])
            return datetime(year, month, day)
        except:
            return None

    def load_latest_data(self):
        """Always load the most recent data"""
        if self.most_recent_date:
            return self.load_data_for_date(self.most_recent_date)
        return pd.DataFrame()

    def load_historical_range(self, days=30):
        """Load last N days of historical data"""
        if not self.all_dates:
            return pd.DataFrame()

        dates_to_load = self.all_dates[-days:] if len(self.all_dates) >= days else self.all_dates

        all_data = []
        for date_str in dates_to_load:
            df = self.load_data_for_date(date_str)
            if not df.empty:
                df['date'] = date_str
                df['date_formatted'] = self.format_date(date_str)
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


# Initialize the enhanced processor
processor = EnhancedDataProcessor()

# Main Header
st.markdown('<h1 class="main-header">📈 NSE Swing Trading Intelligence System</h1>', unsafe_allow_html=True)

# Status Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<span class="present-indicator">LIVE: {processor.format_date(processor.most_recent_date) if processor.most_recent_date else "No Data"}</span>',
        unsafe_allow_html=True)
with col2:
    st.caption(f"📊 Total Days: **{len(processor.all_dates)}**")
with col3:
    st.caption(f"👤 User: **TechPeacock**")
with col4:
    st.caption(f"⏰ Time: **{datetime.now().strftime('%I:%M %p')}**")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    if processor.all_dates:
        # Data Summary
        st.success(f"✅ Database: {len(processor.all_dates)} days")

        oldest_date = processor.format_date(processor.all_dates[0])
        newest_date = processor.format_date(processor.all_dates[-1])

        st.info(f"""
        📅 **Data Coverage**
        First: {oldest_date}
        Latest: {newest_date}
        """)

        # Analysis Settings
        st.markdown("### ⚙️ Analysis Settings")

        # Historical days for comparison
        historical_days = st.slider(
            "Historical days for analysis",
            min_value=5,
            max_value=min(60, len(processor.all_dates)),
            value=min(30, len(processor.all_dates)),
            help="Number of historical days to use for trend analysis"
        )

        # Filters
        st.markdown("### 🔍 Signal Filters")
        min_confidence = st.slider("Min Confidence %", 30, 100, 60)
        min_delivery = st.slider("Min Delivery %", 0, 100, 50)
        max_stocks_display = st.slider("Max stocks to show", 10, 50, 20)

        # Signal type
        signal_filter = st.selectbox(
            "Signal Type",
            ["All Signals", "BUY Only", "SELL Only", "HOLD Only"]
        )

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    else:
        st.error("❌ No data found in ./data/")
        st.stop()

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Today's Signals",
    "📊 Market Overview",
    "🔬 Stock Analysis",
    "📈 Historical Trends",
    "📚 Help Guide"
])

# Tab 1: Today's Signals (PRESENT FOCUSED)
with tab1:
    st.markdown(f"## 🎯 Trading Signals for {processor.format_date(processor.most_recent_date)}")

    # Generate signals button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Generate Latest Signals", type="primary", use_container_width=True):
            with st.spinner("Analyzing latest market data..."):
                # Load ONLY the most recent data for signals
                latest_df = processor.load_latest_data()

                if not latest_df.empty:
                    # Apply feature engineering
                    engineer = FeatureEngineer()
                    latest_df = engineer.create_features(latest_df)

                    # Simple signal generation based on delivery and price movement
                    if 'DELIV_PER' in latest_df.columns:
                        # Generate signals
                        latest_df['signal'] = 'HOLD'
                        latest_df['confidence'] = 50.0

                        # BUY signals
                        buy_condition = (latest_df['DELIV_PER'] > 60)
                        latest_df.loc[buy_condition, 'signal'] = 'BUY'
                        latest_df.loc[buy_condition, 'confidence'] = latest_df.loc[buy_condition, 'DELIV_PER']

                        # SELL signals
                        sell_condition = (latest_df['DELIV_PER'] < 40)
                        latest_df.loc[sell_condition, 'signal'] = 'SELL'
                        latest_df.loc[sell_condition, 'confidence'] = 100 - latest_df.loc[sell_condition, 'DELIV_PER']

                        # Store in session state
                        st.session_state['latest_signals'] = latest_df
                        st.session_state['signal_date'] = processor.most_recent_date
                        st.success(f"✅ Generated signals for {len(latest_df)} stocks")

    with col2:
        if st.button("📥 Export CSV", use_container_width=True):
            if 'latest_signals' in st.session_state:
                csv = st.session_state['latest_signals'].to_csv(index=False)
                st.download_button(
                    "Download",
                    csv,
                    f"signals_{processor.most_recent_date}.csv",
                    "text/csv"
                )

    with col3:
        if st.button("📧 Email Report", use_container_width=True):
            st.info("Email feature coming soon!")

    # Display signals
    if 'latest_signals' in st.session_state:
        signals_df = st.session_state['latest_signals']

        # Apply filters
        if min_delivery > 0 and 'DELIV_PER' in signals_df.columns:
            signals_df = signals_df[signals_df['DELIV_PER'] >= min_delivery]

        if min_confidence > 0 and 'confidence' in signals_df.columns:
            signals_df = signals_df[signals_df['confidence'] >= min_confidence]

        if signal_filter != "All Signals":
            signal_map = {"BUY Only": "BUY", "SELL Only": "SELL", "HOLD Only": "HOLD"}
            signals_df = signals_df[signals_df['signal'] == signal_map[signal_filter]]

        # Summary metrics
        st.markdown("### 📊 Signal Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        total_signals = len(signals_df)
        buy_count = (signals_df['signal'] == 'BUY').sum() if 'signal' in signals_df.columns else 0
        sell_count = (signals_df['signal'] == 'SELL').sum() if 'signal' in signals_df.columns else 0
        hold_count = (signals_df['signal'] == 'HOLD').sum() if 'signal' in signals_df.columns else 0

        with col1:
            st.metric("Total", total_signals)
        with col2:
            st.metric("🟢 BUY", buy_count, f"{buy_count / total_signals * 100:.0f}%" if total_signals > 0 else "0%")
        with col3:
            st.metric("🔴 SELL", sell_count, f"{sell_count / total_signals * 100:.0f}%" if total_signals > 0 else "0%")
        with col4:
            st.metric("⚪ HOLD", hold_count, f"{hold_count / total_signals * 100:.0f}%" if total_signals > 0 else "0%")
        with col5:
            avg_conf = signals_df['confidence'].mean() if 'confidence' in signals_df.columns else 0
            st.metric("Avg Conf", f"{avg_conf:.0f}%")

        st.markdown("---")

        # Display top signals
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🟢 Top BUY Signals")
            buy_df = signals_df[signals_df['signal'] == 'BUY'].sort_values('confidence', ascending=False).head(
                max_stocks_display)

            for _, row in buy_df.iterrows():
                st.markdown(f"""
                <div class="stock-card-buy">
                    <b>{row.get('SYMBOL', 'N/A')}</b><br>
                    💰 ₹{row.get('CLOSE_PRICE', 0):.2f} | 
                    📦 {row.get('DELIV_PER', 0):.1f}% | 
                    📊 {row.get('confidence', 0):.0f}%
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔴 Top SELL Signals")
            sell_df = signals_df[signals_df['signal'] == 'SELL'].sort_values('confidence', ascending=False).head(
                max_stocks_display)

            for _, row in sell_df.iterrows():
                st.markdown(f"""
                <div class="stock-card-sell">
                    <b>{row.get('SYMBOL', 'N/A')}</b><br>
                    💰 ₹{row.get('CLOSE_PRICE', 0):.2f} | 
                    📦 {row.get('DELIV_PER', 0):.1f}% | 
                    📊 {row.get('confidence', 0):.0f}%
                </div>
                """, unsafe_allow_html=True)

        with col3:
            st.markdown("### ⚪ Notable HOLD")
            hold_df = signals_df[signals_df['signal'] == 'HOLD'].sort_values('TURNOVER_LACS', ascending=False).head(
                max_stocks_display)

            for _, row in hold_df.iterrows():
                st.markdown(f"""
                <div class="stock-card-hold">
                    <b>{row.get('SYMBOL', 'N/A')}</b><br>
                    💰 ₹{row.get('CLOSE_PRICE', 0):.2f} | 
                    📦 {row.get('DELIV_PER', 0):.1f}%
                </div>
                """, unsafe_allow_html=True)

# Tab 2: Market Overview (PRESENT FOCUSED)
with tab2:
    st.markdown(f"## 📊 Market Overview - {processor.format_date(processor.most_recent_date)}")

    latest_df = processor.load_latest_data()

    if not latest_df.empty:
        # Market statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_stocks = len(latest_df)
            st.metric("Total Stocks", f"{total_stocks:,}")

        with col2:
            if 'TURNOVER_LACS' in latest_df.columns:
                total_turnover = latest_df['TURNOVER_LACS'].sum()
                st.metric("Total Turnover", f"₹{total_turnover:,.0f} L")

        with col3:
            if 'DELIV_PER' in latest_df.columns:
                avg_delivery = latest_df['DELIV_PER'].mean()
                st.metric("Avg Delivery %", f"{avg_delivery:.1f}%")

        with col4:
            if 'TTL_TRD_QNTY' in latest_df.columns:
                total_volume = latest_df['TTL_TRD_QNTY'].sum()
                st.metric("Total Volume", f"{total_volume:,.0f}")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Delivery distribution
            if 'DELIV_PER' in latest_df.columns:
                fig = px.histogram(
                    latest_df,
                    x='DELIV_PER',
                    nbins=50,
                    title="Delivery % Distribution (Today)",
                    labels={'DELIV_PER': 'Delivery %', 'count': 'Number of Stocks'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top delivery stocks
            if 'DELIV_PER' in latest_df.columns and 'SYMBOL' in latest_df.columns:
                top_delivery = latest_df.nlargest(15, 'DELIV_PER')[['SYMBOL', 'DELIV_PER']]
                fig = px.bar(
                    top_delivery,
                    x='DELIV_PER',
                    y='SYMBOL',
                    orientation='h',
                    title="Top 15 High Delivery Stocks (Today)",
                    labels={'DELIV_PER': 'Delivery %'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# Tab 3: Individual Stock Analysis (PRESENT + HISTORICAL)
with tab3:
    st.markdown("## 🔬 Individual Stock Analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        stock_symbol = st.text_input(
            "Enter Stock Symbol",
            value=st.session_state.get('selected_stock', ''),
            placeholder="e.g., RELIANCE, TCS, INFY"
        ).upper()

    with col2:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if stock_symbol and (analyze_btn or stock_symbol != st.session_state.get('selected_stock', '')):
        st.session_state['selected_stock'] = stock_symbol

        # Load historical data for the stock
        historical_df = processor.load_historical_range(historical_days)

        if not historical_df.empty and 'SYMBOL' in historical_df.columns:
            stock_data = historical_df[historical_df['SYMBOL'] == stock_symbol].copy()

            if not stock_data.empty:
                # Get latest data for current metrics
                latest_stock_data = stock_data[stock_data['date'] == processor.most_recent_date]

                if not latest_stock_data.empty:
                    latest = latest_stock_data.iloc[-1]

                    st.markdown(f"### {stock_symbol} - Current Status")
                    st.caption(f"As of: {processor.format_date(processor.most_recent_date)}")

                    # Current metrics
                    col1, col2, col3, col4, col5, col6 = st.columns(6)

                    with col1:
                        st.metric("Price", f"₹{latest.get('CLOSE_PRICE', 0):.2f}")

                    with col2:
                        change = latest.get('CLOSE_PRICE', 0) - latest.get('PREV_CLOSE', 0)
                        change_pct = (change / latest.get('PREV_CLOSE', 1)) * 100
                        st.metric("Change", f"₹{change:.2f}", f"{change_pct:+.2f}%")

                    with col3:
                        st.metric("Delivery %", f"{latest.get('DELIV_PER', 0):.1f}%")

                    with col4:
                        st.metric("Volume", f"{latest.get('TTL_TRD_QNTY', 0):,.0f}")

                    with col5:
                        st.metric("Trades", f"{latest.get('NO_OF_TRADES', 0):,.0f}")

                    with col6:
                        # Signal
                        if latest.get('DELIV_PER', 0) > 60:
                            st.metric("Signal", "🟢 BUY")
                        elif latest.get('DELIV_PER', 0) < 40:
                            st.metric("Signal", "🔴 SELL")
                        else:
                            st.metric("Signal", "⚪ HOLD")

                    st.markdown("---")

                    # Historical chart
                    st.markdown(f"### 📈 {historical_days}-Day Historical Analysis")

                    # Prepare data for plotting
                    stock_data['date_obj'] = stock_data['date'].apply(processor.get_date_object)
                    stock_data = stock_data.sort_values('date_obj')

                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Price Movement", "Volume", "Delivery %")
                    )

                    # Price chart
                    fig.add_trace(
                        go.Candlestick(
                            x=stock_data['date_obj'],
                            open=stock_data['OPEN_PRICE'],
                            high=stock_data['HIGH_PRICE'],
                            low=stock_data['LOW_PRICE'],
                            close=stock_data['CLOSE_PRICE'],
                            name="Price"
                        ),
                        row=1, col=1
                    )

                    # Volume chart
                    fig.add_trace(
                        go.Bar(
                            x=stock_data['date_obj'],
                            y=stock_data['TTL_TRD_QNTY'],
                            name="Volume",
                            marker_color='blue'
                        ),
                        row=2, col=1
                    )

                    # Delivery % chart
                    colors = ['green' if x > 60 else 'red' if x < 40 else 'gray'
                              for x in stock_data['DELIV_PER']]
                    fig.add_trace(
                        go.Bar(
                            x=stock_data['date_obj'],
                            y=stock_data['DELIV_PER'],
                            name="Delivery %",
                            marker_color=colors
                        ),
                        row=3, col=1
                    )

                    fig.update_layout(
                        height=700,
                        showlegend=False,
                        xaxis3_tickformat='%d-%b',
                        xaxis3_title="Date"
                    )

                    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    fig.update_yaxes(title_text="Delivery %", row=3, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                    # Statistics table
                    st.markdown("### 📊 Statistical Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Avg Price", f"₹{stock_data['CLOSE_PRICE'].mean():.2f}")
                        st.metric("Price Volatility", f"{stock_data['CLOSE_PRICE'].std():.2f}")

                    with col2:
                        st.metric("Avg Delivery %", f"{stock_data['DELIV_PER'].mean():.1f}%")
                        st.metric("Max Delivery %", f"{stock_data['DELIV_PER'].max():.1f}%")

                    with col3:
                        st.metric("Avg Volume", f"{stock_data['TTL_TRD_QNTY'].mean():,.0f}")
                        st.metric("Max Volume", f"{stock_data['TTL_TRD_QNTY'].max():,.0f}")

                    with col4:
                        st.metric("Total Days", len(stock_data))
                        high_delivery_days = (stock_data['DELIV_PER'] > 60).sum()
                        st.metric("High Delivery Days", high_delivery_days)

                else:
                    st.warning(f"No data for {stock_symbol} on {processor.format_date(processor.most_recent_date)}")
            else:
                st.error(f"❌ Stock {stock_symbol} not found in database")
        else:
            st.error("No historical data available")

# Tab 4: Historical Trends
with tab4:
    st.markdown("## 📈 Market Historical Trends")

    # Load historical data
    historical_df = processor.load_historical_range(historical_days)

    if not historical_df.empty:
        st.info(f"Analyzing last {historical_days} trading days")

        # Prepare data
        historical_df['date_obj'] = historical_df['date'].apply(processor.get_date_object)

        # Daily aggregates
        daily_stats = historical_df.groupby('date_obj').agg({
            'DELIV_PER': 'mean',
            'TURNOVER_LACS': 'sum',
            'TTL_TRD_QNTY': 'sum',
            'SYMBOL': 'count'
        }).reset_index()
        daily_stats.columns = ['Date', 'Avg_Delivery', 'Total_Turnover', 'Total_Volume', 'Active_Stocks']

        # Trend charts
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                daily_stats,
                x='Date',
                y='Avg_Delivery',
                title='Average Delivery % Trend',
                markers=True
            )
            fig.update_xaxes(tickformat='%d-%b')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                daily_stats,
                x='Date',
                y='Total_Turnover',
                title='Total Market Turnover Trend (Lacs)',
                markers=True
            )
            fig.update_xaxes(tickformat='%d-%b')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Top consistent performers
        st.markdown("### 🏆 Consistent High Delivery Stocks")

        stock_stats = historical_df.groupby('SYMBOL')['DELIV_PER'].agg(['mean', 'std', 'count'])
        stock_stats = stock_stats[stock_stats['count'] >= historical_days * 0.8]  # Present in 80% of days
        top_consistent = stock_stats.sort_values('mean', ascending=False).head(20)

        if not top_consistent.empty:
            top_consistent = top_consistent.reset_index()
            top_consistent.columns = ['Symbol', 'Avg Delivery %', 'Std Dev', 'Days Present']

            st.dataframe(
                top_consistent.style.format({
                    'Avg Delivery %': '{:.1f}%',
                    'Std Dev': '{:.2f}',
                    'Days Present': '{:.0f}'
                }),
                use_container_width=True
            )

# Tab 5: Help Guide
with tab5:
    st.markdown("## 📚 Quick Reference Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Understanding Signals

        **🟢 BUY Signal**
        - Delivery % > 60%
        - Strong institutional buying
        - Good for swing trades (2-5 days)

        **🔴 SELL Signal**
        - Delivery % < 40%
        - High speculation/selling pressure
        - Avoid or exit positions

        **⚪ HOLD Signal**
        - Delivery % between 40-60%
        - Mixed signals
        - Wait for clear direction

        ### 📊 Key Metrics

        **Delivery %**
        - Higher = Stronger hands (investors)
        - Lower = Weaker hands (traders)
        - >70% = Very bullish
        - <30% = Very bearish
        """)

    with col2:
        st.markdown("""
        ### 💡 Trading Tips

        1. **Focus on High Confidence**
           - Only trade signals >70% confidence
           - Higher delivery = Higher confidence

        2. **Check Volume**
           - High volume + High delivery = Strong signal
           - Low volume = Weak signal

        3. **Use Stop Loss**
           - BUY trades: -2% stop loss
           - Target: +3-5% profit

        4. **Best Times**
           - Generate signals after 6 PM
           - Place orders before 9:15 AM
           - Hold for 2-5 days

        ### 🚀 System Features

        - **Real-time Analysis**: Always uses latest data
        - **Historical Comparison**: Up to 2 years of data
        - **Smart Filtering**: Customizable parameters
        - **Export Options**: CSV download available
        """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>NSE Swing Trading Intelligence System v2.0</strong><br>
    Latest Data: {processor.format_date(processor.most_recent_date)} | 
    Database: {len(processor.all_dates)} days | 
    User: TechPeacock<br>
    {datetime.now().strftime('%d-%b-%Y %I:%M %p')}
</div>
""", unsafe_allow_html=True)