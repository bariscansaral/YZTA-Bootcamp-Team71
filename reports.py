from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from text_mapping import MEDICAL_DISCLAIMER, translate_model_results


class ReportGenerator:
    def __init__(self) -> None:
        self.messages = {
            "anemia": {
                "Not Anemia": "Anemi sınıfı yönünden pozitif model çıktısı bulunmadı.",
                "Anemia": "Model, anemi sınıfı yönünde pozitif çıktı üretmiştir.",
            },
            "autoimmune": {
                "Systemic lupus erythematosus (SLE)": (
                    "Model çıktısı SLE sınıfıyla en yüksek benzerliği göstermiştir."
                ),
                "Graves' disease": (
                    "Model çıktısı Graves hastalığı sınıfıyla en yüksek benzerliği göstermiştir."
                ),
                "Rheumatoid arthritis": (
                    "Model çıktısı romatoid artrit sınıfıyla en yüksek benzerliği göstermiştir."
                ),
                "Sjögren syndrome": (
                    "Model çıktısı Sjögren sendromu sınıfıyla en yüksek benzerliği göstermiştir."
                ),
                "Autoimmune orchitis": (
                    "Model çıktısı otoimmün orşit sınıfıyla en yüksek benzerliği göstermiştir."
                ),
                "Other": (
                    "Model çıktısı diğer otoimmün durumlar sınıfına atanmıştır."
                ),
                "None": (
                    "Otoimmün sınıflandırma sonucu bildirilmemiştir."
                ),
            },
        }

    @staticmethod
    def _percent(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0

        if number <= 1:
            number *= 100

        return max(0.0, min(number, 100.0))

    def generate_report(
        self,
        results: Mapping[str, Any],
        reliability_scores: Mapping[str, float],
    ) -> str:
        """Doktor/teknik raporu oluşturur."""

        report = [
            "",
            "=" * 54,
            "NEUROSCAN AI - TEKNİK KARAR DESTEK RAPORU",
            "=" * 54,
            f"Rapor tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        ]

        anemia_res = results.get("anemia", "Bilinmiyor")
        anemia_score = self._percent(
            reliability_scores.get("anemia", 0)
        )

        report.extend(
            [
                "",
                "[ANEMİ MODELİ]",
                f"Teknik sonuç: {anemia_res}",
                self.messages["anemia"].get(
                    anemia_res,
                    "Tanımlanmayan bir model çıktısı alınmıştır.",
                ),
                f"Veri yeterliliği: %{anemia_score:.1f}",
            ]
        )

        autoimmune_res = results.get(
            "autoimmune",
            "Bilinmiyor",
        )

        autoimmune_score = self._percent(
            reliability_scores.get("autoimmune", 0)
        )

        report.extend(
            [
                "",
                "[OTOİMMÜN MODELİ]",
                f"Teknik sonuç: {autoimmune_res}",
                self.messages["autoimmune"].get(
                    autoimmune_res,
                    "Tanımlanmayan bir model çıktısı alınmıştır.",
                ),
                f"Veri yeterliliği: %{autoimmune_score:.1f}",
            ]
        )

        neuro_res = results.get(
            "neuro",
            "Tümör taraması yapılamadı",
        )

        report.extend(
            [
                "",
                "[BEYİN MR MODELİ]",
                f"Teknik sonuç: {neuro_res}",
            ]
        )

        if anemia_score < 70 or autoimmune_score < 70:
            report.extend(
                [
                    "",
                    "[DÜŞÜK VERİ YETERLİLİĞİ UYARISI]",
                    (
                        "Eksik parametreler model sonucunun "
                        "güvenilirliğini azaltabilir."
                    ),
                    (
                        "Eksik kan değerleri tamamlanarak "
                        "analiz tekrarlanmalıdır."
                    ),
                ]
            )

        report.extend(
            [
                "",
                "[MEDİKAL UYARI]",
                (
                    "Bu çıktı klinik tanı değildir; "
                    "eğitim ve karar destek amacı taşır."
                ),
                (
                    "Nihai değerlendirme uzman hekim "
                    "tarafından yapılmalıdır."
                ),
                "=" * 54,
                "",
            ]
        )

        return "\n".join(report)

    def generate_patient_report_data(
        self,
        results: Mapping[str, Any],
        reliability_scores: Mapping[str, float] | None = None,
        patient_name: str | None = None,
    ) -> dict[str, Any]:

        translated = translate_model_results(
            results,
            reliability_scores,
        )

        return {
            "title": "NeuroScan AI - Hasta Bilgilendirme Notu",
            "report_date": datetime.now().strftime(
                "%d.%m.%Y %H:%M"
            ),
            "patient_name": patient_name or "Belirtilmedi",
            "sections": [
                {
                    "title": (
                        "Kan Değerleri - "
                        "Anemi Değerlendirmesi"
                    ),
                    **translated["anemia"],
                },
                {
                    "title": (
                        "Kan Değerleri - "
                        "Otoimmün Değerlendirmesi"
                    ),
                    **translated["autoimmune"],
                },
                {
                    "title": (
                        "Beyin MR Görüntüsü "
                        "Değerlendirmesi"
                    ),
                    **translated["neuro"],
                },
            ],
            "next_steps": [
                (
                    "Sonuçları tek başına tanı olarak "
                    "değerlendirmeyiniz."
                ),
                (
                    "Kan tahlili ve MR sonuçlarınızı "
                    "uzman hekiminizle paylaşınız."
                ),
                (
                    "Eksik kan parametreleri varsa "
                    "analizi tamamlanmış verilerle "
                    "tekrarlayınız."
                ),
            ],
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def generate_patient_report(
        self,
        results: Mapping[str, Any],
        reliability_scores: Mapping[str, float] | None = None,
        patient_name: str | None = None,
    ) -> str:

        data = self.generate_patient_report_data(
            results,
            reliability_scores,
            patient_name,
        )

        lines = [
            "=" * 54,
            data["title"].upper(),
            "=" * 54,
            f"Rapor tarihi: {data['report_date']}",
            f"Hasta: {data['patient_name']}",
        ]

        for section in data["sections"]:
            lines.extend(
                [
                    "",
                    f"[{section['title'].upper()}]",
                    section["patient_text"],
                    section["reliability_text"],
                ]
            )

        lines.extend(
            [
                "",
                "[SONRAKİ ADIMLAR]",
            ]
        )

        lines.extend(
            f"- {item}"
            for item in data["next_steps"]
        )

        lines.extend(
            [
                "",
                "[ÖNEMLİ UYARI]",
                data["disclaimer"],
                "=" * 54,
            ]
        )

        return "\n".join(lines)