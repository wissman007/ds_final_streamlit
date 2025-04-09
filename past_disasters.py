import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from os import path
import folium
import matplotlib.pyplot as plt

# from streamlit_option_menu import option_menu

# Il faudrait mettre les constantes (par exemple les chemins) dans des venv.
# Pour l'instant, je répertoire de travail est la racine du projet.

@st.cache_data
def load_data():
    # df = pd.read_csv(path.join('data', 'train_data.csv'))
    df = pd.read_csv(path.join('data', 'xview_dataset.csv'))
    return df

st.title('Past disaster analysis')

tab1, tab2 = st.tabs(["General information", "Statistics"])

with tab1:
    st.header("Training data")
    # st.dataframe(load_data().rename(columns={load_data().columns[0]: "index"}), use_container_width=True, hide_index=True)
    st.dataframe(load_data(), use_container_width=True, hide_index=True)
    st.header("Localisation")

    fig = px.scatter_geo(data_frame=load_data().dropna(), 
                        lat=load_data().dropna()['lat'],
                        lon=load_data().dropna()['lgt'],
                        color=load_data().dropna()['disaster_type'],
                        projection='equirectangular', 
                        hover_name=load_data().dropna()['disaster'],
                        size=load_data().dropna()['severity'], 
                        animation_frame=load_data().dropna()['moment'],
                        width=1500, height=768)
    st.plotly_chart(fig)

    #Carte géographique des catastrophes (Folium)
    # Création de la carte
    df = load_data()
    df_post = df[df['moment'] == 'post']
    m = folium.Map(location=[df_post['lat'].mean(), 
                    df_post['lgt'].mean()], 
                    zoom_start=4)

    # Ajout des marqueurs
    for idx, row in df_post.dropna().iterrows():
        folium.Marker(
            location=[row['lat'], row['lgt']],
            popup=f"{row['disaster_type']}<br>Sévérité: {row['severity']}",
            icon=folium.Icon(color='red' if row['moment'] == 'post' else 'blue')
        ).add_to(m)
    
    st.components.v1.html(folium.Figure().add_child(m).render(), height=500)
    

with tab2:
    st.header("Disaster repartition")
    
    pie_data = {
        'disaster_type': load_data()['disaster_type'].value_counts().index, 
        'number of events': load_data()['disaster_type'].value_counts()
    }

    pie_df = pd.DataFrame(pie_data)
    
    pie_fig = px.pie(pie_df, values=pie_df['number of events'], 
                     names=pie_df['disaster_type'])
    
    st.plotly_chart(pie_fig)

    #Répartition des types de catastrophes (Matplotlib)

    # Configuration
    fig_charts = plt.figure(figsize=(12, 6))
    disaster_counts = df['disaster_type'].value_counts()

    # Bar plot
    disaster_counts.plot(kind='bar', color='skyblue')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig_charts)