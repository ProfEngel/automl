
from operator import index
from PIL import Image
import streamlit as st
from pycaret.regression import *
from pycaret.classification import *
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import urllib.request
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE
import lightgbm as lgb
lgb.basic_params["verbosity"] = -1

target_column = None

#Seitenparameter
#im = Image.open("https://github.com/ProfEngel/automl/blob/133d475fd8cb5574e682fa57d1d0527535f49a32/favicon.ico")
#im = Image.open("favicon.ico")
st.set_page_config(
    page_title="AutoML by ProfEngel",
    #page_icon=im,
    layout="wide",)


# Seitenleiste
with st.sidebar:
    #st.image("https://github.com/ProfEngel/automl/blob/133d475fd8cb5574e682fa57d1d0527535f49a32/profengel_logo.png")
    #st.image("profengel_logo.png")
    st.title("AutoML by ProfEngel")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Download","Prediction"])
    st.info("Mit diesem Tool kann man einen Datensatz mittels dem AutoML-Framework von Pycaret untersuchen. Dabei können verschiedene Protokolle, Diagramme, Vorverarbeitungsprozesse, diverse Modelle evaluiert und schlussendlich das bestmöglich trainierte Modell heruntergeladen werden. Alles vollautomatisch, bzw. auf Wunsch nach Anpassung.")

# 1. Datensatz beziehen
# Hauptauswahl für den Benutzer
source = st.radio("Möchten Sie einen Datensatz aus einer Onlinequelle verwenden oder einen Datensatz hochladen?", ("Onlinequelle", "Hochladen"))

# Überprüfen Sie, ob die Datei existiert und löschen Sie sie, falls sie existiert
if os.path.exists("dataset.csv"):
  os.remove("dataset.csv")

# Wenn der Benutzer einen Datensatz aus einer Onlinequelle verwenden möchte
if source == "Onlinequelle":
    st.title("Onlinequelle angeben")
    online_source = st.text_input("Bitte geben Sie den direkten Link zum CSV-Datensatz ein:")
    if online_source:
        try:
            urllib.request.urlretrieve(online_source, 'dataset.csv')
            df = pd.read_csv('dataset.csv')
            st.dataframe(df)
        except:
            st.error("Fehler beim Herunterladen oder Lesen der CSV-Datei. Bitte überprüfen Sie den Link.")

# Wenn der Benutzer einen Datensatz hochladen möchte
elif source == "Hochladen":
	st.title("Datensatz hochladen")
	file = st.file_uploader("Bitte laden Sie Ihren Datensatz hoch:")
	if file:
		df = pd.read_csv(file, index_col=None)
		df.to_csv('dataset.csv', index=None)
		st.dataframe(df)

# 2. Profiling
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)


# 4. Modellierung
if choice == "Modelling":

    # Zeige immer den Profiling-Bericht, auch wenn der Benutzer zu "Modelling" wechselt
    if "profile_df" in locals():  # Überprüfe, ob der Profiling-Bericht existiert
        st_profile_report(profile_df)


    # Lassen Sie den Benutzer das Target-Merkmal auswählen
    target_column = st.selectbox("Welches Merkmal soll das Target sein?", df.columns)

    # Zeilen mit fehlenden Werten in der Ziel-Spalte entfernen
    df = df.dropna(subset=[target_column])

   # Checkboxen für jede Spalte (außer Target) anzeigen
    st.subheader("Wählen Sie die Merkmale aus, die Sie in der Vorverarbeitung berücksichtigen möchten:")
    features_to_include = {col: st.checkbox(col, value=True) for col in df.columns if col != target_column}

    # Filtern Sie die Daten basierend auf den Auswahlkriterien des Benutzers
    selected_features = [col for col, include in features_to_include.items() if include]
    df = df[selected_features + [target_column]]

    # Zeige die ersten Zeilen des DataFrames
    st.write(df.head())

    # Lassen Sie den Benutzer zwischen automatischer und manueller Vorverarbeitung wählen
    preprocessing_choice = st.radio("Möchten Sie die Vorverarbeitung automatisch durchführen lassen oder selbst auswählen?", ("Automatisch", "Manuell"))

    if preprocessing_choice == "Automatisch":
        # Prüfen, ob die Anzahl der Merkmale (ohne das Zielmerkmal) größer als 10 ist
        pca = st.checkbox("Principal Component Analysis (PCA) durchführen?", value=True) if len(df.columns) - 1 > 10 else False

        preprocessing_params = {
            'data': df,
            'target': target_column,
            'session_id': 42,
            'imputation_type': 'simple',
            'normalize': True,
            'remove_multicollinearity': True,
            'multicollinearity_threshold': 0.95,
            'polynomial_features': True,
            'feature_selection': True,
            'pca': pca
        }

    else:
        # Manuelle Auswahl der Vorverarbeitungsschritte
        st.subheader("Wählen Sie die gewünschten Vorverarbeitungsschritte:")
        imputation_type = st.selectbox("Art der Imputation:", ["simple", "iterative"], index=0, help="Wählen Sie die Art der Imputation für fehlende Werte.")
        normalize = st.checkbox("Daten normalisieren?", value=True, help="Skaliert die Merkmale, so dass sie eine mittlere Wert von 0 und eine Standardabweichung von 1 haben.")
        remove_multicollinearity = st.checkbox("Multikollinearität entfernen?", value=True, help="Entfernt Merkmale, die eine Korrelation über dem angegebenen Schwellenwert aufweisen.")
        multicollinearity_threshold = st.slider("Schwellenwert für Multikollinearität:", 0.0, 1.0, 0.95, help="Merkmale mit einer Korrelation über diesem Schwellenwert werden entfernt.")
        polynomial_features = st.checkbox("Polynomiale Merkmale hinzufügen?", value=True, help="Erzeugt polynomiale Merkmale bis zum angegebenen Grad.")
        feature_selection = st.checkbox("Merkmal-Auswahl durchführen?", value=True, help="Wendet eine Merkmalauswahl an, um die besten Merkmale zu behalten.")

        # Prüfen, ob die Anzahl der Merkmale (ohne das Zielmerkmal) größer als 10 ist
        pca = st.checkbox("Principal Component Analysis (PCA) durchführen?", value=True) if len(df.columns) - 1 > 10 else False


        preprocessing_params = {
            'data': df,
            'target': target_column,
            'session_id': 42,
            'imputation_type': imputation_type,
            'normalize': normalize,
            'remove_multicollinearity': remove_multicollinearity,
            'multicollinearity_threshold': multicollinearity_threshold,
            'polynomial_features': polynomial_features,
            'feature_selection': feature_selection,
            'pca': pca
        }


# Überprüfen Sie, ob target_column definiert ist und nicht None ist
    if target_column is not None:

        # Bestimmen Sie, ob das Zielmerkmal numerisch oder kategorisch ist
        if isinstance(df[target_column].iloc[0], (int, float)):
            # Wenn das Zielmerkmal kontinuierlich ist, verwende das pycaret.regression Modul
            from pycaret.regression import *
            if st.button("Regressionstraining starten"):
                reg = setup(**preprocessing_params)
                #reg = setup(data=df, target=target_column, session_id=42)
                setup_df=pull()
                st.info("Dies ist das AutoML Training")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df=pull()
                st.info("Hier sind die AutoML Modelle")
                st.dataframe(setup_df)
                st.write(f"Bestes Modell: {best_model}")
                # Diagramme für Regression
                #evaluate_model(best)

                # Zeige die ersten Zeilen des DataFrames
                st.write(df.head())

                # Statistische Zusammenfassung
                st.write(df.describe())

                # Modell vorhersagen
                predictions = predict_model(best_model)
                st.write("Vorhersagen des Modells:")
                st.dataframe(predictions)

                # Modell speichern
                save_model(best_model, 'best_model')
                st.write("Das Modell wurde als 'best_model' gespeichert.")

        else:
            # Wenn das Zielmerkmal kategorisch ist, verwende das pycaret.classification Modul
            from pycaret.classification import *
            from imblearn.over_sampling import RandomOverSampler, SMOTE

            # Prüfen Sie die Anzahl der Beispiele in jeder Klasse
            class_counts = df[target_column].value_counts()

            # Prüfen Sie die Anzahl der Beispiele in der kleinsten Klasse
            min_class_count = class_counts.min()

            # Wenn die kleinste Klasse weniger als 6 Beispiele hat (standardmäßige n_neighbors für SMOTE + 1),
            # verwenden Sie ROS, andernfalls verwenden Sie SMOTE.
            if min_class_count < 6:
                resampling_method = RandomOverSampler()
            else:
                resampling_method = SMOTE()

            preprocessing_params['fix_imbalance'] = True
            preprocessing_params['fix_imbalance_method'] = resampling_method

            # Benutzer wählt das Kriterium aus
            metrics = {
                "AUC": "AUC (Area Under the Curve)",
                "Accuracy": "Accuracy",
                "Recall": "Recall",
                "Precision": "Precision",
                "F1": "F1 Score"
            }
            sort_metric = st.selectbox("Wählen Sie das Kriterium für die Modellauswahl:", list(metrics.keys()))
            st.info(metrics[sort_metric])

            # Hyperparameter-Tuning
            #if st.checkbox("Möchten Sie Hyperparameter-Tuning durchführen?"):
                #tuned_model = tune_model(best_model)
                #st.write(f"Getuntes Modell: {tuned_model}")

            if st.button("Klassifikationstraining starten"):
                clf = setup(**preprocessing_params)
                #clf = setup(data=df, target=target_column, session_id=42)
                setup_df=pull()
                st.info("Dies ist das AutoML Training")
                st.dataframe(setup_df)

                best_model = compare_models(sort=sort_metric)
                st.write(f"Bestes Modell basierend auf {sort_metric}: {best_model}")


                compare_df = pull()
                st.info("Dies ist das ML Modell")
                st.dataframe(compare_df)
                #st.write(f"Bestes Modell: {best_model}")

                # Diagramme für Klassifikation
                #evaluate_model(best)

                # Überprüfen, ob das Modell coef_ oder feature_importances_ Attribute hat
                if hasattr(best_model, 'coef_') or hasattr(best_model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_importance_plot = plot_model(best_model, plot='feature', save=True, verbose=False)
                    st.plotly_chart(feature_importance_plot)
                else:
                    st.warning("Feature Importance ist für dieses Modell nicht verfügbar.")

                # Zeige die ersten Zeilen des DataFrames
                st.write(df.head())

                # Statistische Zusammenfassung
                st.write(df.describe())

                # Modell vorhersagen
                predictions = predict_model(best_model)
                st.write("Vorhersagen des Modells:")
                st.dataframe(predictions)

                # Modell speichern
                save_model(best_model, 'best_model')
                st.write("Das Modell wurde als 'best_model' gespeichert.")

# 5. Dashboard mit Diagrammen und Tabellen
if choice == "Dashboard":
    st.title("Exploratory Data Analysis")

# 6. Download
if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

# 7. Vorhersagen
if choice == "Prediction":
    st.title("Vorhersage auf trainiertes Modell prüfen:")

    # Lade das trainierte Modell
    loaded_model = None
    try:
        loaded_model = load_model('best_model')
    except:
        st.error("Es wurde kein Modell gefunden. Bitte trainieren Sie zuerst ein Modell.")

    # Option für den Benutzer, Daten zum Vorhersagen hochzuladen
    st.subheader("Daten für die Vorhersage hochladen oder eingeben")
    predict_file = st.file_uploader("CSV-Datei mit Daten für die Vorhersage hochladen:")

    if predict_file:
        predict_df = pd.read_csv(predict_file)
        # Stellen Sie sicher, dass die Ziel-Spalte (falls vorhanden) aus dem Vorhersage-DataFrame entfernt wird
        predict_df = predict_df.drop(columns=[target_column], errors='ignore')
        predictions = predict_model(loaded_model, data=predict_df)
        st.subheader("Vorhersageergebnisse")
        st.dataframe(predictions)
    else:
        # Lassen Sie den Benutzer die Daten manuell eingeben, jedoch ohne die Ziel-Spalte
        input_data = {}
        for col in df.columns:
            # Wir überspringen die Ziel-Spalte, da wir diese vorhersagen möchten
            if col != target_column:
                input_data[col] = st.text_input(f"Geben Sie einen Wert für {col} ein:")

        # Überprüfen, ob alle Felder ausgefüllt sind
        if all(val for val in input_data.values()):
            input_df = pd.DataFrame([input_data])
            predictions = predict_model(loaded_model, data=input_df)
            st.subheader("Vorhersageergebnis")
            st.write(predictions[target_column].iloc[0])
