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
