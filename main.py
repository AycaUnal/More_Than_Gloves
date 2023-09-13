import streamlit as st
import pandas as pd
import numpy as np
#from streamlit_lottie import st_lottie
#from streamlit_lottie import st_lottie_spinner
import time
import requests
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split,learning_curve
import xgboost as xgb
import pydeck as pdk

#####################################        STREAMLİT             ##############################################################

#### Sayfa Ayarları #########
st.set_page_config(
    page_title="More Than Gloves",
    page_icon=":gloves:",
    layout="wide",
    initial_sidebar_state="expanded",
    dark_mode = True
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


###### BAŞLIK #######

st.markdown("""
    <h1 style='text-align: center; color: white; font-family: Verdana, sans-serif; font-size: 50px; font-weight: bold; '>
     More Than <span style='color: red; font-style: italic;'>Gloves</span> &#129354;
    </h1>
""", unsafe_allow_html=True)


st.divider()

############ PYTHON #################################
df_ = pd.read_csv("data.csv", sep=',')
df = df_.copy()

######################################################

def fillna_with_mean(df):
    sayisal_sutunlar = df.select_dtypes(include=[np.number])
    sayisal_sutunlar_doldurulmus = sayisal_sutunlar.fillna(sayisal_sutunlar.mean())
    df[sayisal_sutunlar.columns] = sayisal_sutunlar_doldurulmus
    return df

def label_encode_column(df, column_name):
    enc = LabelEncoder()
    df[column_name] = enc.fit_transform(df[column_name])
    return df

def standard_scale_columns(df, columns_to_scale):
    std = StandardScaler()
    df_to_scale = df[columns_to_scale]
    df_scaled = std.fit_transform(df_to_scale)
    df[df_to_scale.columns] = df_scaled
    return df

################ PREPROCESSS###############
def preprocess_data(df):

    opp_iceren_metinler = [metin for metin in df.columns if "_opp_" in metin]
    df.drop(opp_iceren_metinler, axis=1, inplace=True)

    df = df.drop(columns=['R_fighter', 'B_fighter', 'Referee', 'location', 'date', 'title_bout'])


    df = df[df['Winner'] != 'Draw']
    df['Winner'] = df['Winner'].map({'Red': 0, 'Blue': 1})

    df['B_Stance'] = df['B_Stance'].fillna(df['B_Stance'].mode()[0])
    df['R_Stance'] = df['R_Stance'].fillna(df['R_Stance'].mode()[0])

    df = fillna_with_mean(df)

    df = label_encode_column(df, 'weight_class')
    df = label_encode_column(df, 'R_Stance')
    df = label_encode_column(df, 'B_Stance')


    numerical = df.drop(['weight_class', 'Winner','R_Stance','B_Stance'], axis=1)
    columns_to_scale = numerical.select_dtypes(include=[np.float64, np.int64]).columns
    df = standard_scale_columns(df, columns_to_scale)

    return df

df = preprocess_data(df)
df = df.drop(['R_avg_LEG_landed'], axis=1)

with open('egitilmis_xgboost_model_finallll.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


def predict_fight_winner(loaded_model, df, red_fighter, blue_fighter):

    red_data = df[df['R_fighter'] == red_fighter].copy()
    blue_data = df[df['B_fighter'] == blue_fighter].copy()

    list_B = []
    list_R = []

    for i in df.columns:
        if i[0] == 'B':
            list_B.append(i)
        else:
            list_R.append(i)

    print(len(list_B), len(blue_data), len(blue_data[list_B].columns))
    print(len(list_R), len(red_data), len(red_data[list_R].columns))

    blue_data = blue_data[list_B].reset_index()
    red_data = red_data[list_R].reset_index()

    fight_data = pd.concat((blue_data, red_data), axis=1)

    print(len(fight_data.columns), len(fight_data))
    print(fight_data.columns)

    columns_to_drop = ['R_fighter', 'B_fighter', 'Winner']

    df = df.drop(columns_to_drop, axis=1)
    fight_data = fight_data.drop(columns_to_drop, axis=1)[df.columns]


    winner_pred = loaded_model.predict(xgb.DMatrix(fight_data))

    print(len(winner_pred))
    print(winner_pred)


    winner_index = np.argmax(winner_pred)
    if winner_index == 0:
        winner = red_fighter
    else:
        winner = blue_fighter

    return winner



original_fighter_columns = df_[['R_fighter', 'B_fighter']]
final_df = pd.concat([df.reset_index(drop=True), original_fighter_columns.reset_index(drop=True)], axis=1)












#################################################################

####### SIKLET  SEÇİMİ##################
weight_classes = df_['weight_class'].unique()
with st.sidebar:
    st.image('pngwing.com.png',width= 250)

with st.sidebar:
    selected_weight_class = st.select_slider(" _Karşılaşmanın_ _Hangi_ _Sıklette_ _Olacağını_ _Seçiniz_ ", weight_classes)


with st.sidebar:
    st.image('venum-logo.png',width= 250)









#########  Dövüşçü Seçme Kolonu            ###################
filtered_fighters = df_[df_['weight_class'] == selected_weight_class]['R_fighter'].unique().tolist()
filtered_fighters2 = df_[df_['weight_class'] == selected_weight_class]['B_fighter'].unique().tolist()

col1, col2 = st.columns(2)

with col1:
    option = st.selectbox(
    "**_:red[KIRMIZI]_** _**KÖŞEDEKİ**_ _**DÖVÜŞÇÜYÜ**_ _**SEÇİNİZ**_", filtered_fighters)

with col2:
    option2 = st.selectbox(
        "**_:blue[MAVİ]_** _**KÖŞEDEKİ**_ _**DÖVÜŞÇÜYÜ**_ _**SEÇİNİZ**_",filtered_fighters2)
###################################################################################



############# BUTON KISIMLARI #######################################



col1, col2,  = st.columns([1,0.18])

with col1:
    if st.button('Maçı Başlat'):
        nested_col1, nested_col2, nested_col3 = st.columns([1, 2, 0.35])
        with nested_col1:
            st.write(" ")
            #lottie_url2 = "https://assets-v2.lottiefiles.com/a/890cb942-1177-11ee-847c-73f9b2630e61/wskZOKAm5E.json"
            #st_lottie(lottie_url2, width=100, height=100)



        with nested_col2:








            my_bar = st.progress(0.0001)

            for percent_complete in range(100):
                time.sleep(0.03)
                my_bar.progress(percent_complete + 1)

            red_fighter = option  # option ile seçilen kırmızı köşedeki dövüşçü
            blue_fighter = option2  # option2 ile seçilen mavi köşedeki dövüşçü

            winner = predict_fight_winner(loaded_model, final_df, red_fighter, blue_fighter)

            st.success(f"🏆 Maçın Kazananı: {winner}")


            selected_fight_data = df_[(df_['R_fighter'] == red_fighter) & (df_['B_fighter'] == blue_fighter)]

        with nested_col3:
             st.write(" ")
            #lottie_url2 = "https://assets-v2.lottiefiles.com/a/890cb942-1177-11ee-847c-73f9b2630e61/wskZOKAm5E.json"
            #st_lottie(lottie_url2, width=100, height=100)

with col2:
    st.button('Yeni Bir Maç Seç')









########### RESİM EKLEME ###############





col1, col2, col3, col4 = st.columns(4)

with col1:
        st.image("1.jpeg", width=420,use_column_width='auto')#bir

with col2:
        st.image("2.jpeg", width=410,use_column_width='auto')#iki

with col3:
        st.image("3.jpeg", width=370,use_column_width='auto')#7

with col4:
        st.image("4.jpeg", width=400,use_column_width='auto')#6


########### Analizler VE Grafikler###################
#şimdilik rastgele

with st.sidebar:
    on = st.toggle('Takım:sunglasses:')

    if on:
        people = {
         "Raşit Gürses": "https://www.linkedin.com/in/raşit-gürses?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BYni0%2FVRlR1CTQkZhBtRtow%3D%3D",
         "Sema Kocatürk": "https://www.linkedin.com/in/sema-kocatürk?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bb1eLCsTpRaeoh3enjYxzTg%3D%3D",
         "Mustafa Sincer": "https://www.linkedin.com/in/mustafa-sincer-17a1b3167?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BAOIvl5OURTW2pgTfRtzB8w%3D%3D",
         "Ayça Sevil Ünal": "https://www.linkedin.com/in/ayça-sevil-ünal?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BT53q4nQtRM%2BqfImuuy7HSA%3D%3D"
           }
        selected_person = st.radio("",list(people.keys()))


        if selected_person:
            linkedin_link = people[selected_person]
            st.info(selected_person)
            st.info(f"LinkedIn Bağlantısı: [{linkedin_link}]({linkedin_link})")







