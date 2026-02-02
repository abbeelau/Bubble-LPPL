"""
NDX LPPL (Log-Periodic Power Law) Bubble Analysis
Estimates critical time windows for potential market corrections
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NDX LPPL Bubble Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    /* Main text colors - bright and readable */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #f0f0f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #f0f0f0 !important;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: #b0b0b0 !important;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        color: #c0c0c0 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #00d4ff !important;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .warning-box {
        background: rgba(255, 0, 110, 0.15);
        border: 1px solid rgba(255, 0, 110, 0.4);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    .warning-box * {
        color: #ffffff !important;
    }
    
    .info-box {
        background: rgba(0, 212, 255, 0.15);
        border: 1px solid rgba(0, 212, 255, 0.4);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    .info-box * {
        color: #e0e0e0 !important;
    }
    
    .stSidebar {
        background: rgba(10, 10, 15, 0.95);
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        color: #00d4ff !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #d0d0d0 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #b0b0b0 !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #ffffff !important;
    }
    
    /* Table text */
    .stDataFrame * {
        color: #f0f0f0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        color: #ffffff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# LPPL Functions
def lppl(t, A, B, C, m, tc, omega, phi):
    """Log-Periodic Power Law function"""
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)

def residuals(params, t, p):
    """Residuals for least squares optimization"""
    A, B, C, m, tc, omega, phi = params
    if not (0.1 < m < 0.9 and 4 < omega < 15 and t.max() < tc < t.max() + 250):
        return 1e6 * np.ones_like(p)
    return lppl(t, A, B, C, m, tc, omega, phi) - p

def fit_lppl_window(t, p, tc_min_offset=10, tc_max_offset=180):
    """Fit LPPL model to a single time window"""
    A0 = p.mean()
    B0 = -1.0
    C0 = 0.1
    m0 = 0.5
    tc0 = t.max() + (tc_min_offset + tc_max_offset) / 2
    omega0 = 8.0
    phi0 = 0.0
    
    x0 = [A0, B0, C0, m0, tc0, omega0, phi0]
    
    lower = [A0-10, -10, -10, 0.1, t.max()+tc_min_offset, 4, -np.pi]
    upper = [A0+10,  10,  10, 0.9, t.max()+tc_max_offset, 15,  np.pi]
    
    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            args=(t, p),
            max_nfev=2000,
            verbose=0
        )
        return res
    except:
        return None

def scan_tc_distribution(t_all, p_all, lookbacks, step=5, error_threshold=0.01, progress_callback=None):
    """Scan multiple lookback windows to collect tc estimates"""
    tc_list = []
    fits = []
    total_iterations = sum(max(0, (len(t_all) - lb) // step + 1) for lb in lookbacks if lb < len(t_all))
    total_iterations = max(total_iterations, 1)  # Prevent division by zero
    current_iteration = 0
    
    for lb in lookbacks:
        if lb >= len(t_all):
            continue
        for end in range(lb, len(t_all), step):
            start = end - lb
            t_win = t_all[start:end]
            p_win = p_all[start:end]
            
            current_iteration += 1
            if progress_callback:
                progress_callback(min(current_iteration / total_iterations, 1.0))
            
            try:
                res = fit_lppl_window(t_win, p_win)
                if res is not None and res.success:
                    A, B, C, m, tc, omega, phi = res.x
                    err = np.mean(res.fun**2)
                    if err < error_threshold:
                        tc_list.append(tc)
                        fits.append({
                            "start": start,
                            "end": end,
                            "lookback": lb,
                            "tc": tc,
                            "m": m,
                            "omega": omega,
                            "B": B,
                            "C": C,
                            "error": err
                        })
            except Exception:
                continue
    
    return np.array(tc_list), pd.DataFrame(fits)

@st.cache_data(ttl=3600)
def load_data(symbol, start_date):
    """Load and cache price data"""
    data = yf.download(symbol, start=start_date, progress=False)
    return data

def main():
    # Header
    st.markdown('<h1 class="main-title">🔮 NDX LPPL Bubble Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Log-Periodic Power Law Analysis for Critical Time Estimation</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        symbol = st.selectbox(
            "Index Symbol",
            ["^NDX", "^GSPC", "^DJI", "^IXIC", "QQQ", "SPY"],
            index=0,
            help="Select the index to analyze"
        )
        
        lookback_years = st.slider(
            "Data Lookback (Years)",
            min_value=1,
            max_value=5,
            value=2,
            help="How many years of historical data to use"
        )
        
        st.markdown("---")
        st.markdown("### 📊 LPPL Parameters")
        
        lookback_options = st.multiselect(
            "Lookback Windows (Trading Days)",
            options=[60, 90, 120, 180, 252, 378, 504],
            default=[120, 252, 504],
            help="Different time windows to test (60≈3mo, 252≈1yr)"
        )
        
        step_size = st.slider(
            "Scan Step Size",
            min_value=1,
            max_value=20,
            value=5,
            help="Smaller = more thorough but slower"
        )
        
        error_threshold = st.slider(
            "Error Threshold",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Maximum acceptable fitting error"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About LPPL")
        st.markdown("""
        <div class="info-box">
        <small>
        The <b>Log-Periodic Power Law (LPPL)</b> model detects 
        unsustainable super-exponential growth patterns often 
        seen before market corrections.
        <br><br>
        Key parameters:<br>
        • <b>tc</b>: Critical time (potential peak)<br>
        • <b>m</b>: Power law exponent (0.1-0.9)<br>
        • <b>ω</b>: Log-periodic frequency (6-13)
        </small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime("%Y-%m-%d")
    
    # Load data
    with st.spinner("Loading market data..."):
        data = load_data(symbol, start_date)
    
    if data.empty:
        st.error("Failed to load data. Please try again.")
        return
    
    # Handle both single and multi-level column names
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][symbol.replace("^", "")].dropna() if symbol.replace("^", "") in data["Close"].columns else data["Close"].iloc[:, 0].dropna()
    else:
        close = data["Close"].dropna()
    
    t_all = np.arange(len(close))
    p_all = np.log(close.values.flatten())
    
    # Display current data info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${close.iloc[-1]:,.2f}",
            f"{((close.iloc[-1] / close.iloc[-2]) - 1) * 100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Data Points",
            f"{len(close):,}",
            f"{lookback_years} years"
        )
    
    with col3:
        ytd_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
        st.metric(
            "Period Return",
            f"{ytd_return:+.1f}%",
            "Total"
        )
    
    with col4:
        volatility = close.pct_change().std() * np.sqrt(252) * 100
        st.metric(
            "Annualized Vol",
            f"{volatility:.1f}%",
            "Daily"
        )
    
    st.markdown("---")
    
    # Run analysis button
    if st.button("🚀 Run LPPL Analysis", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct):
            progress_bar.progress(pct)
            status_text.text(f"Scanning windows... {pct*100:.0f}%")
        
        with st.spinner("Running LPPL analysis..."):
            tc_arr, fits_df = scan_tc_distribution(
                t_all, p_all,
                lookbacks=lookback_options,
                step=step_size,
                error_threshold=error_threshold,
                progress_callback=update_progress
            )
        
        progress_bar.empty()
        status_text.empty()
        
        if len(tc_arr) == 0:
            st.warning("No valid LPPL fits found. Try adjusting parameters (larger error threshold or different lookback windows).")
            return
        
        # Convert tc indices to dates
        dates = data.index
        tc_dates = []
        for tc in tc_arr:
            idx = int(min(max(round(tc), 0), len(dates)-1))
            if tc > len(dates) - 1:
                # Extrapolate future date
                days_ahead = int(tc - (len(dates) - 1))
                future_date = dates[-1] + pd.Timedelta(days=days_ahead * 1.4)  # Approx calendar days
                tc_dates.append(future_date)
            else:
                tc_dates.append(dates[idx])
        
        tc_dates = pd.Series(tc_dates)
        
        # Results section
        st.markdown("## 📊 Analysis Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Statistics
            st.markdown("### Critical Time Distribution")
            
            q20 = tc_dates.quantile(0.2)
            q50 = tc_dates.quantile(0.5)
            q80 = tc_dates.quantile(0.8)
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("20th Percentile", q20.strftime("%Y-%m-%d"))
            with stats_col2:
                st.metric("Median (50th)", q50.strftime("%Y-%m-%d"))
            with stats_col3:
                st.metric("80th Percentile", q80.strftime("%Y-%m-%d"))
            
            st.markdown(f"""
            <div class="warning-box">
            <b>⚠️ Critical Window Estimate</b><br>
            <span style="font-size: 1.2rem; color: #ff006e;">
            {q20.strftime("%Y-%m-%d")} → {q80.strftime("%Y-%m-%d")}
            </span><br>
            <small>Based on {len(tc_arr)} valid LPPL fits</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Model parameter statistics
            st.markdown("### Model Parameters")
            
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                st.metric("Avg m (exponent)", f"{fits_df['m'].mean():.3f}")
            with param_col2:
                st.metric("Avg ω (frequency)", f"{fits_df['omega'].mean():.2f}")
            with param_col3:
                st.metric("Avg Error", f"{fits_df['error'].mean():.4f}")
        
        with col2:
            # Histogram of tc dates
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=tc_dates,
                nbinsx=25,
                marker_color='rgba(0, 212, 255, 0.6)',
                marker_line_color='rgba(0, 212, 255, 1)',
                marker_line_width=1
            ))
            
            # Add vertical lines for percentiles using shapes (avoiding plotly bug)
            for q, label, color in [(q20, '20%', '#7b2cbf'), (q50, '50%', '#ff006e'), (q80, '80%', '#7b2cbf')]:
                q_str = q.isoformat() if hasattr(q, 'isoformat') else str(q)
                fig_hist.add_shape(
                    type="line",
                    x0=q_str, x1=q_str,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color=color, width=2, dash="dash")
                )
                fig_hist.add_annotation(
                    x=q_str,
                    y=1,
                    yref="paper",
                    text=label,
                    showarrow=False,
                    yshift=10,
                    font=dict(color=color, size=12)
                )
            
            fig_hist.update_layout(
                title="Distribution of Estimated Critical Times (tc)",
                xaxis_title="Date",
                yaxis_title="Frequency",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="JetBrains Mono, monospace"),
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Price chart with LPPL overlay
        st.markdown("### Price Chart with Critical Window")
        
        fig_price = go.Figure()
        
        # Price line
        fig_price.add_trace(go.Scatter(
            x=data.index,
            y=close.values,
            mode='lines',
            name='Price',
            line=dict(color='#00d4ff', width=2)
        ))
        
        # Add shaded region for critical window using shape
        q20_str = q20.isoformat() if hasattr(q20, 'isoformat') else str(q20)
        q50_str = q50.isoformat() if hasattr(q50, 'isoformat') else str(q50)
        q80_str = q80.isoformat() if hasattr(q80, 'isoformat') else str(q80)
        
        fig_price.add_shape(
            type="rect",
            x0=q20_str, x1=q80_str,
            y0=0, y1=1,
            yref="paper",
            fillcolor="rgba(255, 0, 110, 0.15)",
            line_width=0,
            layer="below"
        )
        fig_price.add_annotation(
            x=q20_str,
            y=1,
            yref="paper",
            text="Critical Window",
            showarrow=False,
            xanchor="left",
            yshift=10,
            font=dict(color="#ff006e", size=12)
        )
        
        # Median line using shape
        fig_price.add_shape(
            type="line",
            x0=q50_str, x1=q50_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="#ff006e", width=2, dash="dash")
        )
        fig_price.add_annotation(
            x=q50_str,
            y=0.95,
            yref="paper",
            text="Median tc",
            showarrow=False,
            xanchor="left",
            xshift=5,
            font=dict(color="#ff006e", size=11)
        )
        
        fig_price.update_layout(
            title=f"{symbol} Price with LPPL Critical Window",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="JetBrains Mono, monospace"),
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Detailed fits table
        with st.expander("📋 View Detailed Fit Results"):
            display_df = fits_df.copy()
            display_df['start_date'] = display_df['start'].apply(lambda x: dates[x].strftime("%Y-%m-%d"))
            display_df['end_date'] = display_df['end'].apply(lambda x: dates[min(x, len(dates)-1)].strftime("%Y-%m-%d"))
            display_df['tc_date'] = tc_dates.values
            display_df['tc_date'] = display_df['tc_date'].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
            
            st.dataframe(
                display_df[['start_date', 'end_date', 'lookback', 'tc_date', 'm', 'omega', 'error']].round(4),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"lppl_results_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Interpretation guide
        st.markdown("---")
        st.markdown("### 📖 Interpretation Guide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Window Concentration**
            - Tight window (< 2 months): Strong signal
            - Wide window (> 4 months): Weak signal
            - Multiple clusters: Uncertainty
            """)
        
        with col2:
            st.markdown("""
            **Parameter Quality**
            - m near 0.3-0.5: Typical bubble
            - ω near 6-8: Standard oscillation
            - Low error (< 0.005): High confidence
            """)
        
        with col3:
            st.markdown("""
            **Trading Implications**
            - Reduce leverage near tc
            - Consider protective puts
            - Monitor for acceleration
            """)
        
        st.markdown("""
        <div class="info-box">
        <b>⚠️ Disclaimer:</b> LPPL is a statistical model, not a prediction tool. 
        Critical time estimates indicate periods of elevated risk, not guaranteed turning points. 
        Always combine with other analysis methods and proper risk management.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
