import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("🏘️ Property Finder & Recommender")

location_df = joblib.load("model/location_distance.pkl")
cosine_sim1 = joblib.load("model/cosine_sim1.pkl")
cosine_sim2 = joblib.load("model/cosine_sim2.pkl")
cosine_sim3 = joblib.load("model/cosine_sim3.pkl")


def recommend_properties_with_scores(property_name, top_n):
    cosine_sim_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8 * cosine_sim3
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
    top_properties = location_df.index[top_indices].tolist()
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    return recommendations_df


st.subheader("🔍 Search Properties by Distance")
selected_location = st.selectbox('Select Base Location', sorted(location_df.columns.tolist()))
radius = st.number_input('Radius (Km)', min_value=0.1, max_value=100.0, value=5.0)

if st.button('Search'):
    meter_radius = radius * 1000
    valid_df = location_df[location_df[selected_location].notna()]
    filtered_df = valid_df[valid_df[selected_location] <= meter_radius][selected_location].sort_values()

    if filtered_df.empty:
        st.warning("⚠ No properties found within this radius.")
    else:
        result_df = pd.DataFrame({
            "PropertyName": filtered_df.index,
            "Distance (Km)": (filtered_df.values / 1000).round(1)
        }).reset_index(drop=True)

        st.success(f"🏡 Found {len(result_df)} properties!")
        st.dataframe(result_df)

st.subheader("🏢 Find Similar Apartments")
selected_apartment = st.selectbox('Select an Apartment', sorted(location_df.index.tolist()))

if st.button('Find Similar Apartments'):
    recommendations_df = recommend_properties_with_scores(selected_apartment, 10)
    st.dataframe(recommendations_df)
