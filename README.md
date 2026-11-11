# Comprensión visual de escenas - TPI de Inteligencia Artificial 2026

Pipeline que transforma una imagen en una representación estructurada, verificable y consultable:

```
imagen -> detección -> segmentación -> SceneJSON -> preguntas y respuestas
```

Trabajo Práctico Integrador de la cátedra de Inteligencia Artificial (UTN, Facultad Regional Córdoba), organizado como una competencia privada de Kaggle entre el 2 de septiembre y el 1 de noviembre de 2026.

**Resultado: 2.º puesto entre 86 equipos (326 participantes), con un PipelineScore de 0,7656 en el leaderboard privado.**

## Qué hace

Para cada imagen produce un SceneJSON con los objetos de las 120 categorías del TPI (caja, máscara y confianza), sus atributos derivados de la máscara (centroide, área relativa, zona de la imagen y tamaño) y las relaciones espaciales entre ellos. Sobre ese SceneJSON responde preguntas de dos maneras: un modo determinista y un modo LLM que interpreta texto libre.

## Pipeline

```
imagen
├─ Grounding DINO base              (vocabulario abierto, 120 categorías)
├─ EVA-02-L LVIS+O365: imagen y 2x2 (detector supervisado; también da máscaras)
├─ EVA-02-L LVIS                    (segunda opinión)
└─ SAM 3                            (cajas >= 2 % de la imagen)
   ↓ todos los candidatos juntos
re-puntaje aprendido: P(acierto) por candidato (HistGradientBoosting + Extra Trees)
   ↓ supresión de duplicados (NMS) y umbral
máscaras: EVA-02 en objetos chicos, SAM 3 en medianos y grandes, SAM 2.1 como respaldo
   ↓
SceneJSON: atributos desde la máscara + relaciones espaciales
   ↓
preguntas: modo determinista y modo LLM (Qwen3-1.7B interpreta, el motor responde)
```

Cada decisión (fuentes, modelo de re-puntaje, umbrales, máscaras, relaciones) se toma con validación por mitades: se elige en una mitad de las 120 imágenes de desarrollo y se mide en la otra.

## Evolución del puntaje

| Versión | Cambio principal | Local (desarrollo) | Público (Kaggle) |
|---|---|---|---|
| 3 | Grounding DINO + OWLv2, re-puntaje de candidatos | 0,664 | - |
| 7 | Recortes 2x2, máscaras ampliadas para objetos chicos | 0,668 | - |
| 8 | ViTDet-H entrenado en LVIS como detector principal | 0,7548 | 0,7927 |
| 11 | Soft-NMS, máscaras de SAM 3, relaciones "pareja" | 0,7615 | 0,8028 |
| 12 | Mezcla de clasificadores (HGB + Extra Trees) | 0,7738 | 0,8115 |
| 13 | EVA-02-L en lugar de ViTDet-H | 0,7930 | 0,8255 |

Métrica de la competencia: `PipelineScore = 0,40 x F1 de cajas + 0,35 x F1 de máscaras + 0,15 x consistencia estructurada + 0,10 x QA`.

## Tecnologías

Python, PyTorch, Hugging Face Transformers, Detectron2, scikit-learn, pycocotools. Ejecutado en Kaggle con 2 GPU T4.

## Contenido

- `tpi_ia_2026_comprension_visual.ipynb`: notebook completo, de punta a punta (análisis exploratorio, detección, re-puntaje, segmentación, SceneJSON, preguntas, evaluación y envío).

No se incluyen el dataset de la competencia ni la métrica oficial, que pertenecen a la cátedra. Por eso el notebook no corre tal cual fuera de la competencia: queda como referencia del código y de los resultados.

## Equipo (Grupo 4 - 5K3A)

- Segundo de la Quintana
- Facundo Aranas
- Bautista Garcia
- Marcelo Madrid
- Fernando Vazques
