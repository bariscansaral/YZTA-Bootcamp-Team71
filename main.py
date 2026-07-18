import joblib
from  tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from reports import ReportGenerator

class ModelManager():
    def __init__(self):
        self.anemia=joblib.load("models/anemia_model.pkl")
        self.autoimmune=joblib.load("models/autoimmune_model.pkl")
        self.neuro=load_model("models/neuroscan_mobilenetv2_v2.keras")
        
        self.auto_map={0:"Autoimmune orchitis",1:"Graves' disease",2:"Other",3:"Rheumatoid arthritis",
                       4:"Sjögren syndrome",5:"Systemic lupus erythematosus (SLE)"}
        self.anemia_map={0:"Not Anemia",1:"Anemia"}

        self.required_cols = {
            "anemia": ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"],
            "autoimmune": ["Age", "Gender", "RBC_Count", "Hemoglobin", "Hematocrit", "MCV",
                           "MCH", "MCHC", "RDW", "Reticulocyte_Count", "WBC_Count",
                           "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils",
                           "Basophils", "PLT_Count", "MPV", "CRP", "ESR"]
        }

    def predict_all(self,patient_df,image=None):
        df = patient_df.copy()
        results={}
        reliability = {}

        for model_name, model in [("anemia",self.anemia),("autoimmune",self.autoimmune)]:
            missing = [col for col in self.required_cols[model_name] if col not in df.columns]
            total_cols = len(self.required_cols[model_name])
            reliability[model_name] = (total_cols - len(missing)) / total_cols
            if missing:
                print(f"Uyarı: {model_name} için eksik sütunlar bulundu ve 0 ile dolduruldu: {missing}")
                for col in missing:
                    df[col] = 0

            pred = model.predict(df[self.required_cols[model_name]])[0]

            if model_name == "anemia":
                results["anemia"] = self.anemia_map[pred]
            else:
                results["autoimmune"] = self.auto_map[pred]

        results["neuro"] = "Tümör taraması yapılamadı" if image is None else "Görüntü işlendi..."
        return results, reliability

manager = ModelManager()
test_data_dict = {
    'Age': [30], 'Gender': [0], 'RBC_Count': [4.5], 'Hemoglobin': [13.5],
    'Hematocrit': [40], 'MCV': [80], 'MCH': [30], 'MCHC': [33], 'RDW': [12],
    'Reticulocyte_Count': [1], 'WBC_Count': [7], 'Neutrophils': [50],
    'Lymphocytes': [30], 'Monocytes': [5], 'Eosinophils': [2], 'Basophils': [1],
    'PLT_Count': [200], 'MPV': [10], 'CRP': [1], 'ESR': [10]
}

test_data_eksik = {
    'Gender': [0],
    'Hemoglobin': [13.5],
    'MCV': [80]
}

test_df_tam = pd.DataFrame(test_data_dict)
print("--- Tam Veri Sonucu ---")
print(manager.predict_all(test_df_tam))

test_df_eksik = pd.DataFrame(test_data_eksik)
print("\n--- Eksik Veri Sonucu ---")
print(manager.predict_all(test_df_eksik))

final_results, scores = manager.predict_all(test_df_tam)
reporter = ReportGenerator()
print(reporter.generate_report(final_results, scores))