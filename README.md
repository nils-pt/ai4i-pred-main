# Predictive Maintenance — Explorative Analyse (AI4I 2020)

Explorative Datenanalyse eines Predictive-Maintenance-Datensatzes mit Python, pandas und seaborn. Ziel ist es, Ausfallmuster von Industriemaschinen zu verstehen und die Daten für späteres Machine Learning vorzubereiten.

Dies ist ein Lernprojekt im Rahmen meines Einstiegs in Data Science / Machine Learning.

## Datensatz

- **Quelle:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
- **Umfang:** 10.000 Maschinenzyklen, 14 Spalten
- **Inhalt:** Sensorwerte (Luft- und Prozesstemperatur, Drehzahl, Drehmoment, Werkzeugverschleiß), Produkttyp, ein Hauptlabel (`Machine failure`) sowie fünf spezifische Ausfalltypen (TWF, HDF, PWF, OSF, RNF)

## Fragestellungen & Erkenntnisse

1. **Klassen-Imbalance** — Nur etwa 3,4% der Zyklen führen zu einem Ausfall. Das macht reine Genauigkeit (Accuracy) als Bewertungsmaß ungeeignet: Ein Modell, das immer "kein Ausfall" vorhersagt, hätte bereits 96,6% Genauigkeit, ohne einen einzigen Ausfall zu erkennen.

2. **Stärkste Einflussgrößen** — Werkzeugverschleiß und Drehmoment unterscheiden sich am deutlichsten zwischen Ausfällen und Normalbetrieb.

3. **Ausfälle in Extrembereichen** — Im Scatter-Plot von Drehzahl gegen Drehmoment zeigt sich, dass Ausfälle vor allem an den Rändern auftreten, nicht im Normalbetrieb. Das Problem ist nichtlinear.

4. **Risikoprofile durch Binning** — Nach Einteilung in Kategorien (niedrig/mittel/hoch) steigt die Ausfallrate bei hohem Drehmoment auf rund 18%, gegenüber etwa 1,4% im mittleren Bereich.

5. **Interaktionseffekt** — Bei gleichzeitig hohem Drehmoment UND hohem Werkzeugverschleiß steigt die Ausfallrate auf rund 40% — deutlich mehr als die Summe der Einzelrisiken. Solche Wechselwirkungen kann eine reine Korrelationsanalyse nicht erfassen, was den Bedarf an Machine-Learning-Modellen begründet.

6. **Produktqualität als Faktor** — Produkte niedriger Qualität (Type L) fallen etwa doppelt so häufig aus wie solche hoher Qualität (Type H).

## Vorbereitung für Machine Learning

Ein zentraler Teil der Analyse ist die Feature-Auswahl:

- **Nutzbare Features:** Produkttyp, Sensorwerte (Temperaturen, Drehzahl, Drehmoment, Werkzeugverschleiß) und die abgeleitete Temperatur-Differenz
- **Zielvariable:** `Machine failure`
- **Ausgeschlossen wegen fehlender Vorhersagekraft:** IDs (UDI, Product ID)
- **Ausgeschlossen wegen Data Leakage:** Die spezifischen Ausfalltypen (TWF, HDF, PWF, OSF, RNF). Sie verraten das Ziel direkt und wären zum Vorhersagezeitpunkt gar nicht verfügbar.

## Verwendete Methoden

- Explorative Datenanalyse mit pandas (`groupby`, `value_counts`, Aggregationen)
- Korrelationsanalyse und Heatmaps
- Feature Engineering (Temperatur-Differenz, Binning mit `pd.cut`)
- Visualisierung mit matplotlib und seaborn (Histogramme, Boxplots, Scatter-Plots, Heatmaps)
- Risikoprofile durch zweidimensionale Gruppierung

## Tools

Python · pandas · NumPy · matplotlib · seaborn · Jupyter Notebook

## Dateien

- `predictive_maintenance_explore.ipynb` — das Analyse-Notebook
- `ai4i2020.csv` — der Rohdatensatz

## Ausführen

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook predictive_maintenance_explore.ipynb
```

## Konzepte, die dieses Projekt demonstriert

Klassen-Imbalance · Data Leakage · Interaktionseffekte · nichtlineare Zusammenhänge · Feature-Auswahl · der Unterschied zwischen Korrelation und kombiniertem Risiko
