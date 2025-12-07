import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.title('📊 Real Estate Analytics Dashboard')

new_df = pd.read_csv('data/data_viz1.csv')
feature_text = joblib.load('model/feature_text.pkl')

st.header('📍 Sector-wise Price Per Sqft (Geo Map)')

group_df = (
    new_df
    .groupby("sector")[['price','price_per_sqft','built_up_area','latitude','longitude']]
    .mean()
    .reset_index()
)

fig = px.scatter_mapbox(
    group_df,
    lat="latitude",
    lon="longitude",
    color="price_per_sqft",
    size="built_up_area",
    color_continuous_scale=px.colors.cyclical.IceFire,
    zoom=10,
    mapbox_style="open-street-map",
    height=600,
    hover_name="sector"
)
st.plotly_chart(fig, use_container_width=True)

st.header("📝 Feature Wordcloud")

wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords={"s"},
).generate(feature_text)

fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

st.header('🏠 Built-up Area vs Price')

property_type = st.selectbox('Select property type', ['flat', 'house'])
filtered_df = new_df[new_df['property_type'] == property_type]

fig_area = px.scatter(
    filtered_df,
    x="built_up_area",
    y="price",
    color="bedRoom",
    labels={"bedRoom": "BHK"},
    height=500
)
st.plotly_chart(fig_area, use_container_width=True)

st.header('🥧 BHK Distribution')

sector_options = ['Overall'] + sorted(new_df['sector'].unique().tolist())
selected_sector = st.selectbox('Select Sector', sector_options)

df_pie = new_df if selected_sector == 'Overall' else new_df[new_df['sector'] == selected_sector]
fig_pie = px.pie(df_pie, names='bedRoom', hole=0.3)
st.plotly_chart(fig_pie, use_container_width=True)

st.header('📦 BHK-wise Price Comparison')

fig_box = px.box(
    new_df[new_df['bedRoom'] < 4],
    x='bedRoom',
    y='price',
    labels={"bedRoom": "BHK"},
    height=500
)
st.plotly_chart(fig_box, use_container_width=True)

st.header('📈 Price Distribution by Property Type')

fig_dist, ax_dist = plt.subplots(figsize=(12, 6))

sns.histplot(
    new_df[new_df['property_type'] == 'house']['price'],
    kde=True,
    label='House',
    color='blue',
    ax=ax_dist
)

sns.histplot(
    new_df[new_df['property_type'] == 'flat']['price'],
    kde=True,
    label='Flat',
    color='orange',
    ax=ax_dist
)

ax_dist.legend()
ax_dist.set_xlabel("Price")
ax_dist.set_ylabel("Frequency")

st.pyplot(fig_dist)
