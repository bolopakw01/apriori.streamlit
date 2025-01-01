import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="Association",
    page_icon="🛒",
    layout="wide"
)

# Check if splash screen has been shown
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False

# Splash screen
if not st.session_state.splash_shown:
    splash = st.empty()  # Temporary container for the splash screen
    with splash.container():
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column;">
            <h1 style="color: #FF4B4B; font-size: 3rem; font-weight: bold; text-align: center;">🏪Rekomendasi Keranjang Belanja🏬</h1>
            <p style="font-size: 1.5rem; color: #6c757d; text-align: center;">Memuat aplikasi...</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(3)  # Durasi splash screen

    # Remove the splash screen after 3 seconds
    splash.empty()
    st.session_state.splash_shown = True

# Custom CSS
st.markdown("""
<style>
    /* Card-like containers */
    .stSelectbox, .stSlider {
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    
    h2 {
        color: #FF4B4B;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    
    /* Success message styling */
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00FF00;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .metric-container {
            background-color: rgba(255, 255, 255, 0.05);
        }
    }
</style>
""", unsafe_allow_html=True)

# App title with
st.header("App Analyze")
st.subheader("Association Rule Learning - Apriori Algorithm")
st.title("---------------------------------------------------")
st.markdown("### 📊 Analisis Asosiasi Produk")


# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Main content
    df = pd.read_csv('bread basket.csv')
    df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
    df["month"] = df["date_time"].dt.month
    df["day"] = df["date_time"].dt.day
    
    df["month"].replace([i for i in range(1, 12 + 1)], 
                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                    inplace=True)
    df["day"].replace([i for i in range(6 + 1)], 
                    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                    inplace=True)

# Filter controls in an expander
with st.expander("🔍 Filter Options", expanded=True):
    col3, col4, col5 = st.columns(3)
    
    with col3:
        period_day = st.selectbox('🕒 Period of Day', 
                                ['Morning', 'Afternoon', 'Evening', 'Night'])
    with col4:
        weekday_weekend = st.selectbox('📅 Day Type', 
                                    ['Weekday', 'Weekend'])
    with col5:
        month = st.select_slider('📆 Month', 
                            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

day = st.select_slider('📅 Hari', 
                    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                    value="Sat")

# Item selection with search
ITEMS = ['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral water', 'Farm House', 'Fudge', 'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 'Frittata', 'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies', 'Cake', 'Mighty Protein', 'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray', 'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles', 'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings', 'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup', 'Panatone', 'Brioche and salami', 'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch', 'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck egg', 'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates', 'Coffee granules ', 'Drinking chocolate spoons ', 'Christmas common', 'Argentina Night', 'Half slice Monster ', 'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars', 'Tacos/Fajita']

item = st.selectbox('🔎 Select Item', ITEMS, help="Select an item to analyze")

# Analysis functions remain the same
def get_dataset(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    data["month"] = data["month"].astype(str)
    data["day"] = data["day"].astype(str)
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) &
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] > 0 else "No Result"

def encode(x):
    return 1 if x >= 1 else 0

# Process data
data = get_dataset(period_day.lower(), weekday_weekend.lower(), month, day)

if isinstance(data, pd.DataFrame):
    # Create pivot table
    item_count = data.groupby(['Transaction', 'Item'])["Item"].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)
    
    # Generate rules
    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
    
    if len(frequent_items) > 0:
        rules = association_rules(frequent_items, metric="lift", min_threshold=1, num_itemsets=len(frequent_items))
        rules.sort_values('confidence', ascending=False, inplace=True)

        # Display recommendations in a nice card
        st.markdown("### 🎯 Hasil Rekomendasi")
        matching_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
        
        if not matching_rules.empty:
            consequent = list(matching_rules.iloc[0]['consequents'])[0]
            confidence = matching_rules.iloc[0]['confidence']
            lift = matching_rules.iloc[0]['lift']
            
            # Translate month and day to Indonesian
            month_translation = {
                'Jan': 'Januari', 'Feb': 'Februari', 'Mar': 'Maret', 'Apr': 'April', 'May': 'Mei', 'Jun': 'Juni',
                'Jul': 'Juli', 'Aug': 'Agustus', 'Sep': 'September', 'Oct': 'Oktober', 'Nov': 'November', 'Dec': 'Desember'
            }
            day_translation = {
                'Mon': 'Senin', 'Tue': 'Selasa', 'Wed': 'Rabu', 'Thu': 'Kamis', 'Fri': 'Jumat', 'Sat': 'Sabtu', 'Sun': 'Minggu'
            }
            
            month_ind = month_translation.get(month, month)
            day_ind = day_translation.get(day, day)
            
            st.markdown(f"""            
            <div class="success-message">
                <h4>Dari data yang ada, Customer yang membeli <span style="color: #FF4B4B">{item}</span></h4>
                <h3>Juga akan membeli <span style="color: #FF4B4B">{consequent}</span></h3>
                <p>Lift Score: {lift:.2f}</p>
                <p>Di hari <span style="color: #FF4B4B">{day_ind}</span> bulan <span style="color: #FF4B4B">{month_ind}</span>
                customer yang membeli <span style="color: #FF4B4B">{item}</span> 
                biasanya juga akan membeli <span style="color: #FF4B4B">{consequent}</span> 
                dengan prediksi sebesar <span style="color: #FF4B4B">{confidence*100:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add visualization
            st.markdown("### 📈 Top Associated Items")
            top_rules = matching_rules.head(5)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[list(x)[0] for x in top_rules['consequents']],
                    y=top_rules['confidence'],
                    marker_color='#FF4B4B'
                )
            ])
            
            fig.update_layout(
                title="Confidence Scores for Top Associations",
                xaxis_title="Associated Items",
                yaxis_title="Confidence Score",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada asosiasi kuat yang ditemukan untuk item ini")
    else:
        st.warning("Tidak ada kumpulan item yang sering ditemukan dengan pada batas support saat ini.")
else:
    st.error("Tidak ada data yang tersedia untuk filter yang dipilih")

# Visualisasi distribusi transaksi per hari
st.markdown("### 📊 Distribusi Transaksi per Hari")
fig_day = px.histogram(df, x='day', title='Distribusi Transaksi per Hari')
st.plotly_chart(fig_day, use_container_width=True)

# Visualisasi distribusi transaksi per bulan
st.markdown("### 📊 Distribusi Transaksi per Bulan")
fig_month = px.histogram(df, x='month', title='Distribusi Transaksi per Bulan')
st.plotly_chart(fig_month, use_container_width=True)

# Export data ke CSV
if st.button('Export Hasil Analisis ke CSV'):
    csv = matching_rules.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='hasil_analisis.csv', mime='text/csv')

# Tampilkan detail item yang dipilih
st.markdown("### 📋 Detail Item yang Dipilih")
item_details = df[df['Item'] == item]
st.write(item_details.describe())

# Rekomendasi produk
st.markdown("### 🛒 Rekomendasi Produk")
if not matching_rules.empty:
    recommended_items = matching_rules['consequents'].explode().unique()
    st.write("Produk yang direkomendasikan untuk dibeli bersama:", recommended_items)
else:
    st.write("Tidak ada rekomendasi produk yang ditemukan.")
