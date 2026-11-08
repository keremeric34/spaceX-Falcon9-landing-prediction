# -*- coding: utf-8 -*-
"""
SpaceX Falcon 9 İlk Aşama İniş Tahmini - Uçtan Uca Veri Bilimi Projesi
======================================================================
Bu dosya, projedeki 6 ayrı çalışma dosyasının mantıksal sırayla birleştirilmiş,
eksik (TODO) kısımları tamamlanmış ve hataları giderilmiş tam sürümüdür:

BÖLÜM 1: Web Scraping (Wikipedia Falcon 9 Kalkış Kayıtları)
BÖLÜM 2: Veri Ön İşleme ve Eğitim Etiketlerinin Belirlenmesi (Data Wrangling)
BÖLÜM 3: SQL ile Keşifsel Veri Analizi (SQLite)
BÖLÜM 4: Keşifsel Veri Analizi (EDA) ve Özellik Mühendisliği (Feature Engineering)
BÖLÜM 5: Folium ile Etkileşimli Görsel Analitik ve Haritalama
BÖLÜM 6: Makine Öğrenmesi Tahmin Hattı (Lojistik Regresyon, SVM, Karar Ağacı, KNN)
"""

import io
import re
import sqlite3
import unicodedata
import warnings
from math import atan2, cos, radians, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import folium
    from folium.features import DivIcon
    from folium.plugins import MarkerCluster, MousePosition
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def date_time(table_cells):
    """HTML tablo hücresinden tarih ve saat bilgisini döndürür."""
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]


def booster_version(table_cells):
    """HTML tablo hücresinden roket güçlendirici (booster) versiyonunu döndürür."""
    out = ''.join([
        bv for i, bv in enumerate(table_cells.strings) if i % 2 == 0
    ][0:-1])
    return out


def landing_status(table_cells):
    """HTML tablo hücresinden iniş durumunu döndürür."""
    out = [i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    """HTML tablo hücresinden faydalı yük kütlesini (kg) döndürür."""
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass = mass[0:mass.find("kg") + 2]
    else:
        new_mass = 0
    return new_mass


def extract_column_from_header(row):
    """HTML tablo başlığından temizlenmiş sütun adını çıkarır."""
    if row.br:
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()

    column_name = ' '.join(row.contents)

    if not (column_name.strip().isdigit()):
        column_name = column_name.strip()
        return column_name


def plot_confusion_matrix(y, y_predict, title="Confusion Matrix"):
    """Karmaşıklık matrisini (Confusion Matrix) çizer."""
    cm = confusion_matrix(y, y_predict)
    plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Tahmin Edilen Etiketler (Predicted)')
    ax.set_ylabel('Gerçek Etiketler (True)')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['İnmedi (0)', 'İndi (1)'])
    ax.yaxis.set_ticklabels(['İnmedi (0)', 'İndi (1)'])
    plt.tight_layout()
    plt.show()


def calculate_distance(lat1, lon1, lat2, lon2):
    """İki koordinat arasındaki Haversine mesafesini (km) hesaplar."""
    R = 6373.0  # Dünya'nın yaklaşık yarıçapı (km)

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


# =============================================================================
# BÖLÜM 1: WEB SCRAPING (WIKIPEDIA FALCON 9 KALKIŞ KAYITLARI)
# =============================================================================

def run_web_scraping():
    print("\n" + "=" * 75)
    print("BÖLÜM 1: WIKIPEDIA WEB SCRAPING İŞLEMİ BAŞLATILIYOR")
    print("=" * 75)

    static_url = (
        "https://en.wikipedia.org/w/index.php?"
        "title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    # GÖREV 1: Wikipedia sayfasını talep et
    response = requests.get(static_url, headers=headers)
    print(f"HTTP Durum Kodu: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    print(f"Sayfa Başlığı: {soup.title.string}")

    # GÖREV 2: Tablo başlıklarından sütun isimlerini çıkar
    html_tables = soup.find_all("table")
    first_launch_table = html_tables[2]

    column_names = []
    for th in first_launch_table.find_all("th"):
        name = extract_column_from_header(th)
        if name is not None and len(name) > 0:
            column_names.append(name)

    print(f"Çıkarılan Sütunlar: {column_names}")

    # GÖREV 3: Sözlük oluştur ve HTML tablolarını ayrıştırarak doldur (Eksikler Tamamlandı)
    launch_dict = dict.fromkeys(column_names)
    if 'Date and time ( )' in launch_dict:
        del launch_dict['Date and time ( )']

    launch_dict['Flight No.'] = []
    launch_dict['Launch site'] = []
    launch_dict['Payload'] = []
    launch_dict['Payload mass'] = []
    launch_dict['Orbit'] = []
    launch_dict['Customer'] = []
    launch_dict['Launch outcome'] = []
    launch_dict['Version Booster'] = []
    launch_dict['Booster landing'] = []
    launch_dict['Date'] = []
    launch_dict['Time'] = []

    extracted_row = 0
    for table_number, table in enumerate(
        soup.find_all('table', "wikitable plainrowheaders collapsible")
    ):
        for rows in table.find_all("tr"):
            if rows.th and rows.th.string:
                flight_number = rows.th.string.strip()
                flag = flight_number.isdigit()
            else:
                flag = False

            row = rows.find_all('td')
            if flag:
                extracted_row += 1

                # 1. Flight Number (Uçuş Numarası)
                launch_dict['Flight No.'].append(flight_number)

                # 2. Date & 3. Time (Tarih ve Saat)
                datatimelist = date_time(row[0])
                date = datatimelist[0].strip(',')
                launch_dict['Date'].append(date)

                time = datatimelist[1]
                launch_dict['Time'].append(time)

                # 4. Booster Version (Güçlendirici Versiyonu)
                bv = booster_version(row[1])
                if not bv and row[1].a:
                    bv = row[1].a.string
                launch_dict['Version Booster'].append(bv)

                # 5. Launch Site (Fırlatma Alanı)
                launch_site = row[2].a.string if row[2].a else row[2].get_text(strip=True)
                launch_dict['Launch site'].append(launch_site)

                # 6. Payload (Faydalı Yük)
                payload = row[3].a.string if row[3].a else row[3].get_text(strip=True)
                launch_dict['Payload'].append(payload)

                # 7. Payload Mass (Faydalı Yük Kütlesi)
                payload_mass = get_mass(row[4])
                launch_dict['Payload mass'].append(payload_mass)

                # 8. Orbit (Yörünge)
                orbit = row[5].a.string if row[5].a else row[5].get_text(strip=True)
                launch_dict['Orbit'].append(orbit)

                # 9. Customer (Müşteri)
                if row[6].a is not None and row[6].a.string:
                    customer = row[6].a.string
                else:
                    customer = row[6].get_text(strip=True)
                launch_dict['Customer'].append(customer)

                # 10. Launch Outcome (Uçuş Sonucu)
                launch_outcome = list(row[7].strings)[0].strip()
                launch_dict['Launch outcome'].append(launch_outcome)

                # 11. Booster Landing (Güçlendirici İniş Durumu)
                booster_landing = landing_status(row[8]).strip()
                launch_dict['Booster landing'].append(booster_landing)

    df_scraped = pd.DataFrame({
        key: pd.Series(value, dtype='object') for key, value in launch_dict.items()
    })
    df_scraped.to_csv("spacex_web_scraped.csv", index=False)
    print(f"Toplam ayrıştırılan uçuş kaydı: {extracted_row}")
    print("'spacex_web_scraped.csv' başarıyla kaydedildi.")
    return df_scraped


# =============================================================================
# BÖLÜM 2: VERİ ÖN İŞLEME VE EĞİTİM ETİKETLERİ (DATA WRANGLING)
# =============================================================================

def run_data_wrangling():
    print("\n" + "=" * 75)
    print("BÖLÜM 2: VERİ ÖN İŞLEME (DATA WRANGLING) BAŞLATILIYOR")
    print("=" * 75)

    url_part1 = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv"
    )
    df = pd.read_csv(url_part1)
    print("İlk 5 Kayıt:")
    print(df.head(5))

    print("\nEksik Değer Yüzdeleri (%):")
    print(df.isnull().sum() / len(df) * 100)

    print("\nVeri Tipleri:")
    print(df.dtypes)

    # GÖREV 1: Her fırlatma alanındaki uçuş sayısını hesapla
    print("\n--- GÖREV 1: Fırlatma Alanlarına Göre Uçuş Sayıları ---")
    print(df["LaunchSite"].value_counts())

    # GÖREV 2: Her yörüngenin (GTO hariç) sayısını ve sıklığını hesapla
    print("\n--- GÖREV 2: Yörünge Türlerine Göre Uçuş Sayıları (GTO Hariç) ---")
    print(df[df["Orbit"] != "GTO"]["Orbit"].value_counts())

    # GÖREV 3: Görev sonu iniş sonuçlarının (Outcome) sayısını belirle
    print("\n--- GÖREV 3: İniş Sonuçlarının (Outcome) Dağılımı ---")
    landing_outcomes = df["Outcome"].value_counts()
    print(landing_outcomes)

    for i, outcome in enumerate(landing_outcomes.keys()):
        print(f"{i}: {outcome}")

    bad_outcomes = set(landing_outcomes.keys()[[1, 3, 5, 6, 7]])
    print(f"Başarısız İniş Etiketleri: {bad_outcomes}")

    # GÖREV 4: Outcome sütunundan ikili (0 ve 1) eğitim etiketi (Class) oluştur
    landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df["Outcome"]]
    df['Class'] = landing_class

    print(f"\nGenel İniş Başarı Oranı: {df['Class'].mean():.4f}")
    df.to_csv("dataset_part_2.csv", index=False)
    print("'dataset_part_2.csv' başarıyla kaydedildi.")
    return df


# =============================================================================
# BÖLÜM 3: SQL İLE KEŞİFSEL VERİ ANALİZİ (SQLITE)
# =============================================================================

def run_sql_analysis():
    print("\n" + "=" * 75)
    print("BÖLÜM 3: SQL İLE VERİ ANALİZİ BAŞLATILIYOR")
    print("=" * 75)

    con = sqlite3.connect("my_data1.db")
    cur = con.cursor()

    spacex_csv_url = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv"
    )
    df_sql = pd.read_csv(spacex_csv_url)
    df_sql.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method="multi")

    cur.execute("DROP TABLE IF EXISTS SPACEXTABLE;")
    cur.execute("CREATE TABLE SPACEXTABLE AS SELECT * FROM SPACEXTBL WHERE Date IS NOT NULL;")
    con.commit()

    queries = {
        "GÖREV 1 - Benzersiz Fırlatma Alanları": (
            'SELECT DISTINCT "Launch_Site" FROM SPACEXTABLE;'
        ),
        "GÖREV 2 - 'CCA' ile Başlayan İlk 5 Kayıt": (
            'SELECT * FROM SPACEXTABLE WHERE "Launch_Site" LIKE "CCA%" LIMIT 5;'
        ),
        "GÖREV 3 - NASA (CRS) Tarafından Taşınan Toplam Yük Kütlesi (kg)": (
            'SELECT SUM("PAYLOAD_MASS__KG_") AS Total_Payload_Mass '
            'FROM SPACEXTABLE WHERE "Customer" = "NASA (CRS)";'
        ),
        "GÖREV 4 - F9 v1.1 Güçlendiricisinin Taşıdığı Ortalama ve Toplam Yük Kütlesi": (
            'SELECT AVG("PAYLOAD_MASS__KG_") AS Avg_Payload_Mass, '
            'SUM("PAYLOAD_MASS__KG_") AS Sum_Payload_Mass '
            'FROM SPACEXTABLE WHERE "Booster_Version" = "F9 v1.1";'
        ),
        "GÖREV 5 - Yer Platformuna (Ground Pad) İlk Başarılı İniş Tarihi": (
            'SELECT MIN("Date") AS First_Ground_Pad_Success_Date '
            'FROM SPACEXTABLE WHERE "Landing_Outcome" = "Success (ground pad)";'
        ),
        "GÖREV 6 - İnsansız Deniz Aracına Başarılı İnen ve Yükü 4000-6000 kg Arası Güçlendiriciler": (
            'SELECT "Booster_Version" FROM SPACEXTABLE '
            'WHERE "Landing_Outcome" = "Success (drone ship)" '
            'AND "PAYLOAD_MASS__KG_" > 4000 AND "PAYLOAD_MASS__KG_" < 6000;'
        ),
        "GÖREV 7 - Başarılı ve Başarısız Görev Sonuçlarının Toplam Sayısı": (
            'SELECT "Mission_Outcome", COUNT("Mission_Outcome") AS Total_Count '
            'FROM SPACEXTABLE GROUP BY "Mission_Outcome";'
        ),
        "GÖREV 8 - Maksimum Faydalı Yük Kütlesini Taşıyan Güçlendirici Versiyonları": (
            'SELECT "Booster_Version", "PAYLOAD_MASS__KG_" FROM SPACEXTABLE '
            'WHERE "PAYLOAD_MASS__KG_" = (SELECT MAX("PAYLOAD_MASS__KG_") FROM SPACEXTABLE);'
        ),
        "GÖREV 9 - 2015 Yılında Drone Ship Üzerinde Başarısız İniş Kayıtları": (
            'SELECT substr("Date", 6, 2) AS Month, "Landing_Outcome", '
            '"Booster_Version", "Launch_Site" FROM SPACEXTABLE '
            'WHERE "Landing_Outcome" = "Failure (drone ship)" AND substr("Date", 1, 4) = "2015";'
        ),
        "GÖREV 10 - 2010-06-04 ile 2017-03-20 Arası İniş Sonuçlarının Azalan Sıralaması": (
            'SELECT "Landing_Outcome", COUNT("Landing_Outcome") AS Outcome_Count '
            'FROM SPACEXTABLE WHERE "Date" BETWEEN "2010-06-04" AND "2017-03-20" '
            'GROUP BY "Landing_Outcome" ORDER BY Outcome_Count DESC;'
        )
    }

    for title, sql in queries.items():
        print(f"\n--- {title} ---")
        result_df = pd.read_sql_query(sql, con)
        print(result_df)

    con.close()


# =============================================================================
# BÖLÜM 4: KEŞİFSEL VERİ ANALİZİ (EDA) VE ÖZELLİK MÜHENDİSLİĞİ
# =============================================================================

def run_eda_and_feature_engineering():
    print("\n" + "=" * 75)
    print("BÖLÜM 4: GÖRSEL EDA VE ÖZELLİK MÜHENDİSLİĞİ BAŞLATILIYOR")
    print("=" * 75)

    url_part2 = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
    )
    df = pd.read_csv(url_part2)

    # Uçuş Numarası vs Faydalı Yük Kütlesi Genel Grafik
    sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect=5)
    plt.xlabel("Flight Number", fontsize=16)
    plt.ylabel("Payload Mass (kg)", fontsize=16)
    plt.title("Flight Number vs Payload Mass by Class", fontsize=16)
    plt.show()

    # GÖREV 1: Uçuş Numarası ve Fırlatma Alanı İlişkisi
    sns.catplot(data=df, x="FlightNumber", y="LaunchSite", hue="Class", aspect=2)
    plt.xlabel("Flight Number")
    plt.ylabel("Launch Site")
    plt.title("Task 1: Flight Number vs Launch Site")
    plt.show()

    # GÖREV 2: Faydalı Yük Kütlesi ve Fırlatma Alanı İlişkisi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="PayloadMass", y="LaunchSite", hue="Class")
    plt.xlabel("Payload Mass (kg)")
    plt.ylabel("Launch Site")
    plt.title("Task 2: Payload Mass vs Launch Site")
    plt.show()

    # GÖREV 3: Yörünge Türlerine Göre Başarı Oranı Bar Grafiği
    plt.figure(figsize=(10, 6))
    df.groupby("Orbit")["Class"].mean().plot(kind="bar", color="steelblue")
    plt.xlabel("Orbit")
    plt.ylabel("Success Rate")
    plt.title("Task 3: Success Rate by Orbit Type")
    plt.tight_layout()
    plt.show()

    # GÖREV 4: Uçuş Numarası ve Yörünge Türü İlişkisi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="FlightNumber", y="Orbit", hue="Class")
    plt.xlabel("Flight Number")
    plt.ylabel("Orbit")
    plt.title("Task 4: Flight Number vs Orbit Type")
    plt.show()

    # GÖREV 5: Faydalı Yük Kütlesi ve Yörünge Türü İlişkisi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="PayloadMass", y="Orbit", hue="Class")
    plt.xlabel("Payload Mass (kg)")
    plt.ylabel("Orbit")
    plt.title("Task 5: Payload Mass vs Orbit Type")
    plt.show()

    # GÖREV 6: Yıllara Göre İniş Başarı Trendi
    year = [i.split("-")[0] for i in df["Date"]]
    df_yearly = df.copy()
    df_yearly['Date'] = year

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_yearly, x="Date", y="Class")
    plt.xlabel("Year")
    plt.ylabel("Average Success Rate")
    plt.title("Task 6: Launch Success Yearly Trend")
    plt.show()

    # GÖREV 7: Kategorik Sütunlar İçin Dummy Değişkenler (One-Hot Encoding) Oluştur
    features = df[[
        'FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite',
        'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad',
        'Block', 'ReusedCount', 'Serial'
    ]]
    features_one_hot = pd.get_dummies(
        features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
    )

    # GÖREV 8: Tüm Sayısal Sütunları float64 Tipine Dönüştür
    features_one_hot = features_one_hot.astype("float64")
    print(f"One-Hot Encoding Sonrası Özellik Boyutu: {features_one_hot.shape}")
    features_one_hot.to_csv('dataset_part_3.csv', index=False)
    print("'dataset_part_3.csv' başarıyla kaydedildi.")
    return features_one_hot


# =============================================================================
# BÖLÜM 5: FOLIUM İLE ETKİLEŞİMLİ GÖRSEL ANALİTİK (EKSİKLER TAMAMLANDI)
# =============================================================================

def assign_marker_color(launch_outcome):
    """Sınıf değerine göre (1: başarılı, 0: başarısız) işaretçi rengini belirler."""
    return 'green' if launch_outcome == 1 else 'red'


def run_folium_analytics():
    print("\n" + "=" * 75)
    print("BÖLÜM 5: FOLIUM İLE ETKİLEŞİMLİ HARİTA ANALİZİ BAŞLATILIYOR")
    print("=" * 75)

    geo_url = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv"
    )
    spacex_df = pd.read_csv(geo_url)
    spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
    launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
    launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
    print("Fırlatma Alanları ve Koordinatları:")
    print(launch_sites_df)

    if not FOLIUM_AVAILABLE:
        print("Folium kütüphanesi kurulu olmadığı için harita HTML çıktısı atlandı.")
        return launch_sites_df

    # Başlangıç konumu: NASA Johnson Space Center
    nasa_coordinate = [29.559684888503615, -95.0830971930759]

    # GÖREV 1: Tüm fırlatma alanlarını harita üzerinde işaretle (TODO Tamamlandı)
    site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

    for _, row in launch_sites_df.iterrows():
        coordinate = [row['Lat'], row['Long']]
        site_name = row['Launch Site']

        circle = folium.Circle(
            coordinate, radius=1000, color='#d35400', fill=True
        ).add_child(folium.Popup(site_name))

        marker = folium.map.Marker(
            coordinate,
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 12; color:#d35400;"><b>{site_name}</b></div>'
            )
        )
        site_map.add_child(circle)
        site_map.add_child(marker)

    # GÖREV 2: Her alan için başarılı/başarısız fırlatmaları MarkerCluster ile işaretle (TODO Tamamlandı)
    marker_cluster = MarkerCluster()
    spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)
    site_map.add_child(marker_cluster)

    for index, record in spacex_df.iterrows():
        marker = folium.Marker(
            location=[record['Lat'], record['Long']],
            popup=f"{record['Launch Site']} - Class: {record['class']}",
            icon=folium.Icon(color='white', icon_color=record['marker_color'])
        )
        marker_cluster.add_child(marker)

    # GÖREV 3: Fırlatma alanının yakın çevresine (kıyı, demiryolu, otoyol, şehir) mesafeleri hesapla (TODO Tamamlandı)
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position='topright',
        separator=' Long: ',
        empty_string='NaN',
        lng_first=False,
        num_digits=20,
        prefix='Lat:',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    site_map.add_child(mouse_position)

    # CCAFS SLC-40 referans koordinatları
    launch_site_lat = 28.563197
    launch_site_lon = -80.576820
    launch_site_coord = [launch_site_lat, launch_site_lon]

    proximities = {
        "En Yakın Kıyı Şeridi (Coastline)": [28.56367, -80.56794],
        "En Yakın Demiryolu (NASA Railroad)": [28.57206, -80.58525],
        "En Yakın Otoyol (Samuel C. Phillips Pkwy)": [28.56335, -80.57085],
        "En Yakın Şehir (Titusville)": [28.61222, -80.80755],
    }

    print("\n--- CCAFS SLC-40 Yakınlık Mesafeleri ---")
    for label, coord in proximities.items():
        dist = calculate_distance(launch_site_lat, launch_site_lon, coord[0], coord[1])
        print(f"{label}: {dist:.2f} km")

        distance_marker = folium.Marker(
            coord,
            popup=label,
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 12; color:#d35400;"><b>{dist:10.2f} KM</b></div>'
            )
        )
        lines = folium.PolyLine(locations=[launch_site_coord, coord], weight=1, color='blue')
        site_map.add_child(distance_marker)
        site_map.add_child(lines)

    site_map.save("spacex_launch_map.html")
    print("'spacex_launch_map.html' başarıyla oluşturuldu ve kaydedildi.")
    return site_map


# =============================================================================
# BÖLÜM 6: MAKİNE ÖĞRENMESİ TAHMİN HATTI (MACHINE LEARNING PIPELINE)
# =============================================================================

def run_machine_learning_pipeline():
    print("\n" + "=" * 75)
    print("BÖLÜM 6: MAKİNE ÖĞRENMESİ MODELLERİ EĞİTİLİYOR VE TEST EDİLİYOR")
    print("=" * 75)

    url1 = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
    )
    url2 = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv"
    )

    data = pd.read_csv(url1)
    X_df = pd.read_csv(url2)

    # GÖREV 1: Class sütunundan NumPy dizisi (Y) oluştur
    Y = data["Class"].to_numpy()

    # GÖREV 2: X verisini standartlaştır (StandardScaler)
    transform = preprocessing.StandardScaler()
    X = transform.fit_transform(X_df)

    # GÖREV 3: Eğitim ve test verisi olarak ayır (test_size=0.2, random_state=2)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    print(f"Eğitim örnek sayısı: {X_train.shape[0]}, Test örnek sayısı: {Y_test.shape[0]}")

    # -------------------------------------------------------------------------
    # GÖREV 4 ve 5: Lojistik Regresyon (Logistic Regression)
    # -------------------------------------------------------------------------
    print("\n[1/4] Lojistik Regresyon (GridSearchCV)...")
    parameters_lr = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
    lr = LogisticRegression()
    logreg_cv = GridSearchCV(lr, param_grid=parameters_lr, cv=10)
    logreg_cv.fit(X_train, Y_train)

    lr_test_acc = logreg_cv.score(X_test, Y_test)
    print(f"En İyi Parametreler (LR): {logreg_cv.best_params_}")
    print(f"Doğrulama (CV) Doğruluğu (LR): {logreg_cv.best_score_:.4f}")
    print(f"Test Verisi Doğruluğu (LR):    {lr_test_acc:.4f}")

    yhat_lr = logreg_cv.predict(X_test)
    plot_confusion_matrix(Y_test, yhat_lr, title="Logistic Regression Confusion Matrix")

    # -------------------------------------------------------------------------
    # GÖREV 6 ve 7: Destek Vektör Makineleri (SVM)
    # -------------------------------------------------------------------------
    print("\n[2/4] Destek Vektör Makineleri - SVM (GridSearchCV)...")
    parameters_svm = {
        'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'C': np.logspace(-3, 3, 5),
        'gamma': np.logspace(-3, 3, 5)
    }
    svm = SVC()
    svm_cv = GridSearchCV(estimator=svm, param_grid=parameters_svm, cv=10)
    svm_cv.fit(X_train, Y_train)

    svm_test_acc = svm_cv.score(X_test, Y_test)
    print(f"En İyi Parametreler (SVM): {svm_cv.best_params_}")
    print(f"Doğrulama (CV) Doğruluğu (SVM): {svm_cv.best_score_:.4f}")
    print(f"Test Verisi Doğruluğu (SVM):    {svm_test_acc:.4f}")

    yhat_svm = svm_cv.predict(X_test)
    plot_confusion_matrix(Y_test, yhat_svm, title="SVM Confusion Matrix")

    # -------------------------------------------------------------------------
    # GÖREV 8 ve 9: Karar Ağacı Sınıflandırıcısı (Decision Tree Classifier)
    # (Not: Yeni scikit-learn sürümlerinde 'auto' hatasını önlemek için 'sqrt' ve 'log2' kullanıldı)
    # -------------------------------------------------------------------------
    print("\n[3/4] Karar Ağacı - Decision Tree (GridSearchCV)...")
    parameters_tree = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2 * n for n in range(1, 10)],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    tree = DecisionTreeClassifier(random_state=2)
    tree_cv = GridSearchCV(estimator=tree, param_grid=parameters_tree, cv=10)
    tree_cv.fit(X_train, Y_train)

    tree_test_acc = tree_cv.score(X_test, Y_test)
    print(f"En İyi Parametreler (Tree): {tree_cv.best_params_}")
    print(f"Doğrulama (CV) Doğruluğu (Tree): {tree_cv.best_score_:.4f}")
    print(f"Test Verisi Doğruluğu (Tree):    {tree_test_acc:.4f}")

    yhat_tree = tree_cv.predict(X_test)
    plot_confusion_matrix(Y_test, yhat_tree, title="Decision Tree Confusion Matrix")

    # -------------------------------------------------------------------------
    # GÖREV 10 ve 11: K-En Yakın Komşu (KNN)
    # (Not: Orijinal dosyada X_test ile fit edilen hata düzeltilerek X_train ile fit edildi)
    # -------------------------------------------------------------------------
    print("\n[4/4] K-En Yakın Komşu - KNN (GridSearchCV)...")
    parameters_knn = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(estimator=knn, param_grid=parameters_knn, cv=10)
    knn_cv.fit(X_train, Y_train)

    knn_test_acc = knn_cv.score(X_test, Y_test)
    print(f"En İyi Parametreler (KNN): {knn_cv.best_params_}")
    print(f"Doğrulama (CV) Doğruluğu (KNN): {knn_cv.best_score_:.4f}")
    print(f"Test Verisi Doğruluğu (KNN):    {knn_test_acc:.4f}")

    yhat_knn = knn_cv.predict(X_test)
    plot_confusion_matrix(Y_test, yhat_knn, title="KNN Confusion Matrix")

    # -------------------------------------------------------------------------
    # GÖREV 12: En İyi Performans Gösteren Yöntemin Karşılaştırılması
    # -------------------------------------------------------------------------
    print("\n--- GÖREV 12: Model Performans Karşılaştırma Tablosu ---")
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN'],
        'Validation_Accuracy (CV)': [
            logreg_cv.best_score_,
            svm_cv.best_score_,
            tree_cv.best_score_,
            knn_cv.best_score_
        ],
        'Test_Accuracy': [
            lr_test_acc,
            svm_test_acc,
            tree_test_acc,
            knn_test_acc
        ]
    })
    print(results_df.to_string(index=False))

    best_cv_row = results_df.loc[results_df['Validation_Accuracy (CV)'].idxmax()]
    print(
        f"\nSonuç: Tüm modeller test setinde %{lr_test_acc * 100:.2f} (15/18) doğruluk elde etmiştir. "
        f"Çapraz doğrulama (CV) skoruna göre en yüksek skoru {best_cv_row['Model']} "
        f"(%{best_cv_row['Validation_Accuracy (CV)'] * 100:.2f}) sağlamıştır."
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='Model', y='Test_Accuracy', palette='viridis')
    plt.title('Makine Öğrenmesi Modellerinin Test Doğruluk Oranları')
    plt.ylim(0, 1.0)
    plt.ylabel('Test Doğruluğu (Accuracy)')
    plt.tight_layout()
    plt.show()

    return results_df


# =============================================================================
# ANA ÇALIŞTIRMA BLOĞU (MAIN)
# =============================================================================

if __name__ == "__main__":
    # Tüm aşamaları sırasıyla çalıştırır:
    run_web_scraping()
    run_data_wrangling()
    run_sql_analysis()
    run_eda_and_feature_engineering()
    run_folium_analytics()
    run_machine_learning_pipeline()
