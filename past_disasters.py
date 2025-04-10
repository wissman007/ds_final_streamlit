import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from os import path
import folium
import matplotlib.pyplot as plt
import seaborn as sns

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

    #Carte géographique des catastrophes (Folium)
    # Création de la carte
    df = load_data()
    df_post = df[df['moment'] == 'post']

    fig = px.scatter_geo(data_frame=df_post.dropna(), 
                        lat=df_post.dropna()['lat'],
                        lon=df_post.dropna()['lgt'],
                        color=df_post.dropna()['disaster_type'],
                        projection='equirectangular', 
                        hover_name=df_post.dropna()['disaster'],
                        size=df_post.dropna()['severity'], 
                        animation_frame=df_post.dropna()['moment'],
                        width=1500, height=768)
    st.plotly_chart(fig)

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

    st.header("Severity by disaster type repartition")

    # Filtrer les données post-catastrophe
    df_post = df[df['moment'] == 'post']
    df_post['highest_severity'] = df_post['highest_severity'].replace({
        'un-classified': 'no-damage',
        'Unknown': 'no-damage'
    })
    # Ordre personnalisé et gestion des catégories
    severity_order = [ 'destroyed', 'major-damage', 'minor-damage' ,'no-damage']
    df_post['highest_severity'] = pd.Categorical(
        df_post['highest_severity'],
        categories=severity_order,
        ordered=True
    )
    # Créer un tableau croisé des données et trier par sévérité
    cross_tab = pd.crosstab(
        index=df_post['disaster_type'],
        columns=df_post['highest_severity'],
        normalize='index'
    ).sort_values(by=severity_order, ascending=False)  # Tri ajouté ici
    # Configuration du style
    last_fig = plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    # Création du graphique avec les couleurs personnalisées
    colors = [ 'red',  'orange', 'yellow', 'cyan']
    cross_tab.plot(kind='bar',
                stacked=True,
                color=colors,
                edgecolor='black',
                ax=plt.gca())
    # Personnalisation
    plt.title('Severity by disaster type repartition')
    plt.xlabel('Disaster type')
    plt.ylabel('Severity distribution')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Severity level',
            bbox_to_anchor=(1.05, 1),
            loc='upper left')
    # Ajouter les annotations de pourcentage
    for n, x in enumerate([*cross_tab.index.values]):
        for (proportion, y_loc) in zip(cross_tab.loc[x], cross_tab.loc[x].cumsum()):
            if proportion > 0:
                plt.text(x=n - 0.17,
                        y=(y_loc - proportion) + (proportion / 2),
                        s=f'{proportion:.1%}',
                        color='black',
                        fontsize=8,
                        fontweight='bold')
    plt.tight_layout()
    plt.show()
    st.pyplot(last_fig)