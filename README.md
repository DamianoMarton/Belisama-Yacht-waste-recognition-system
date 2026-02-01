# Belisama Yacht Waste Recognition 🚤🚮

**Synthetic Data Pipeline and YOLO-like Object Detector for Waste Recognition in Venice Workboats.**

Questo progetto sviluppa un sistema di visione artificiale per il riconoscimento automatico dei rifiuti galleggianti nei canali di Venezia. Il sistema è progettato per operare sui nastri trasportatori delle imbarcazioni di **Belisama Yacht**, fornendo report periodici sulla tipologia e quantità di rifiuti raccolti.

---

## 📋 Indice
- [Descrizione del Progetto](#descrizione-del-progetto)
- [Struttura della Repository](#struttura-della-repository)
- [Pipeline del Dataset Sintetico](#pipeline-del-dataset-sintetico)
- [Modelli di Object Detection](#modelli-di-object-detection)
- [Risultati](#risultati)
- [Contatti](#contatti)

---

## 📝 Descrizione del Progetto
A causa della scarsità di dataset reali per i rifiuti acquatici veneziani, abbiamo sviluppato una pipeline per la generazione di **immagini sintetiche realistiche**. Il sistema analizza i fotogrammi estratti dal nastro trasportatore tramite una camera zenitale. 

Il progetto confronta due approcci:
1. **Fine-tuning di YOLOv11 Nano**: per prestazioni professionali.
2. **Custom Minimal YOLO**: un modello da 1.6M di parametri sviluppato da zero per analizzare i meccanismi di detection (grid-based).

---

## 📂 Struttura della Repository

```text
├── dataset/
│   ├── synthetic_scripts/      # Script per la costruzione del dataset sintetico
│   ├── backgrounds/            # Immagini del nastro trasportatore utilizzate come sfondo
│   ├── class_preferences.json  # Parametri (colori, size, rotazione) per ogni classe
│   └── main_reference.json     # Limiti e dimensioni dell'area di lavoro
│
├── scripts/
│   ├── my_yolo/                # Codice del modello custom (Loss, Metriche, Notebook Colab)
│   └── yolo/                   # Notebook per fine-tuning YOLOv11 e script di test
│
├── trained_models/
│   ├── my_yolo/                # Pesi del modello custom (.pth)
│   └── yolo/                   # Pesi del modello YOLOv11 fine-tunato (.pt)
│
└── README.md
```

## 🛠 Pipeline del Dataset Sintetico
La generazione dei dati è gestita dagli script nella cartella dataset/synthetic_scripts/.

**Logica di funzionamento**:
**Configurazione**: I parametri di realismo sono definiti in class_preferences.json (es. probabilità di aderenza al nastro, rotazione, scaling).
**Preprocessing**: Le immagini degli oggetti vengono ritagliate, orientate orizzontalmente e filtrate per variazioni di colore e luminosità.
**Augmentation**: Viene applicato un blur uniforme per ridurre i bordi netti dell'overlay e simulare oscillazioni della telecamera tramite crop dinamici del background definiti in main_reference.json.
Nota: La cartella finale yolo_dataset (~1GB) è esclusa dalla repository per limiti di spazio.

## 🧠 Modelli di Object Detection
**YOLOv11 Nano**
Abbiamo utilizzato il transfer learning per adattare YOLOv11n al nostro scenario specifico (14 classi di rifiuti + 1 classe "other").

**Notebook di training**: Disponibile in scripts/yolo/
**Performance**: Precision 0.97, Recall 0.93.

**Custom Minimal YOLO-like**
Sviluppato per fini di ricerca interna, utilizza:

**Architettura**: Backbone convoluzionale (stride 2) e Neck con blocchi residui (skip connections).
**Grid**: 20x20 cells (fino a 400 predizioni).
**Loss**: Binary Cross Entropy (objectness), IoU Loss (bounding boxes) e Cross Entropy (classi).
**NMS**: Implementazione vettorizzata di Non-Maximum Suppression.
**Codice sorgente**: Disponibile in scripts/my_yolo/

## 📈 Risultati
Il modello custom ha raggiunto un F1-score di 0.69, dimostrando ottime capacità di localizzazione, pur risentendo delle occlusioni pesanti in casi di oggetti sovrapposti. YOLOv11n garantisce invece la robustezza necessaria per l'impiego industriale.

## 🤝 Contatti
**Damiano Marton** - Belisama Yacht
📧 damiano.marton@studenti.unipd.it
📧 damianomarton@belisamayacht.it
🌐 www.belisamayacht.it
