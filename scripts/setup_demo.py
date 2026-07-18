import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import MedicalDB

def run_setup():
    db=MedicalDB()

    doc_id=db.add_user("Dr. Ahmet Acar", 25, is_doctor=True)
    user_id=db.add_user("Samet Aydın", 28, is_doctor=False)
    full_blood_data = {
        "Age": 25,
        "RBC_Count": 4.5,
        "Hemoglobin": 13.5,
        "Hematocrit": 40.0,
        "MCV": 88.0,
        "MCH": 30.0,
        "MCHC": 33.0,
        "RDW": 12.5,
        "Reticulocyte_Count": 1.0,
        "WBC_Count": 7.2,
        "Neutrophils": 60.0,
        "Lymphocytes": 30.0,
        "Monocytes": 5.0,
        "Eosinophils": 4.0,
        "Basophils": 1.0,
        "PLT_Count": 250000,
        "MPV": 9.5,
        "CRP": 2.5,
        "ESR": 10.0
    }
    results = {"anemia": "Not Anemia", "autoimmune": "None"}
    db.add_medical_record(user_id, doc_id, results, full_blood_data)
    print("Tam kan değerleriyle birlikte demo verisi oluşturuldu!")


if __name__ == "__main__":
    run_setup()