class ReportGenerator():
    def __init__(self):
        self.messages={
            "anemia":{
                "Not Anemia":"Anemi bulgusu saptanmadı.",
                "Anemia":"Anemi riski mevcut, bir dahiliye uzmanına başvurunuz."
            },
            "autoimmune":{
                "Systemic lupus erythematosus (SLE)": "SLE bulguları saptandı, detaylı romatolojik tetkik gerekir.",
                "Graves' disease": "Graves hastalığına dair bulgular mevcut, endokrinolojiye danışınız.",
                "Rheumatoid arthritis": "Romatoid artrit bulguları saptandı.",
                "Sjögren syndrome": "Sjögren sendromu ile uyumlu değerler saptandı.",
                "Autoimmune orchitis": "Otoimmün orşit bulguları mevcut, üroloji konsültasyonu önerilir.",
                "Other": "Otoimmün bir durum saptandı, lütfen doktorunuzla görüşün."
            }
        }

    def generate_report(self,results, reliability_scores):
            report=["\n" + "="*40, "NEUROSCAN AI - ANALİZ RAPORU", "="*40]

            anemia_res = results.get("anemia", "Bilinmiyor")
            report.append(f"[+] Anemi Durumu: {anemia_res}")
            report.append(f"    {self.messages['anemia'].get(anemia_res, 'Detaylı inceleme gerekir.')}")

            auto_res = results.get("autoimmune", "Bilinmiyor")
            auto_score = reliability_scores.get("autoimmune", 0) * 100
            report.append(f"\n[!] Otoimmün Durumu: {auto_res}")
            report.append(f"    {self.messages['autoimmune'].get(auto_res, 'Rutin kontrol önerilir.')}")

            if auto_score < 70:
                report.append(f"\n[!] UYARI: Otoimmün analiz güvenilirliği düşük (%{auto_score:.0f}).")
                report.append("    Kesinlik için eksik verilerinizi (CRP, ESR, vb.) sisteme giriniz.")

            report.append("\n" + "=" * 40 + "\n")
            return "\n".join(report)
