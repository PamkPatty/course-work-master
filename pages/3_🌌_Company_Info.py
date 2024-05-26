import streamlit as st
import yfinance as yf
import cufflinks as cf
import datetime
import yahoo_fin.stock_info as si
import base64


st.set_page_config(page_title='Company Info')
# Sidebar
st.sidebar.subheader('Query parameters📦')
ticker_list = si.tickers_sp500()
tickerSymbol = st.sidebar.selectbox('🎫Stock ticker:', ticker_list) # Select ticker symbol

start_date = st.sidebar.date_input("📆Start date:", datetime.date(2013, 1, 1))
end_date = st.sidebar.date_input("🏁End date:", datetime.date(2023, 6, 2))

# Retrieving tickers data

tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# App title
string_name = tickerData.info['longName']
st.title(string_name + "🌌")

# string_logo = '<img src=%s>' % tickerData.info['logo_url']
# st.markdown(string_logo, unsafe_allow_html=True)

def show_company_info(tickerSymbol):
    company = yf.Ticker(tickerSymbol)
    info = company.info
    st.write(f"**🌐Website:** {info['website']}")
    st.write(f"**💹Current Price:** {info['currentPrice']}💲")

    st.title("**About Company**")
    st.write(f"**🚀Sector:** {info['sector']}")
    st.write(f"**🏭Industry:** {info['industry']}")
    st.write(f"**🌎Country:** {info['country']}")
    st.write(f"**🏢City:** {info['city']}")
    st.write(f"**📞Phone:** {info['phone']}")

def main():
    show_company_info(tickerSymbol)

if __name__ == '__main__':
    main()


st.title("**Business Summary🧠**")
company = yf.Ticker(tickerSymbol)
info = company.info
st.write(f"**🤼Full Time Employees:** {info['fullTimeEmployees']}")

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Bollinger bands
st.header('**Bollinger Bands📈**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

# Ticker data
st.header('**Ticker data🎫**')
st.write(tickerDf)

def filedownload(tickerDf):
    csv = tickerDf.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="ticker.csv">📁Download CSV File</a>'
    return href

st.markdown(filedownload(tickerDf), unsafe_allow_html=True)
    

