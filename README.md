# Mitterrand 1981 — Une conversation avec l’histoire

Agent conversationnel historique en français, développé avec **Gradio 6**. Le modèle utilisé est optimisé via LoRA à partir d'un ensemble de 600 déclarations de François Mitterand.

## Lancer l'app

Depuis la racine du projet, avec l’environnement Python contenant Gradio :

```powershell
.venv\Scripts\python.exe app\app.py
```

Ouvrir l’adresse locale indiquée dans le terminal.

Depuis un notebook situé à la racine du projet :

```python
from app.app import launch_app

launch_app(respond=ask)
```

`respond(question, history) -> str` est la fonction qui interroge le modèle. `history` contient les 5 derniers échanges au format `[{"role": ..., "content": str}]`. Sans `respond`, l'app reste en mode démonstration. Utiliser `launch_app()` pour charger également le CSS avec Gradio 6. Les chemins des ressources sont résolus à partir du fichier Python.

Avec le modèle fine-tuné, `notebooks/chat_mitterrand.ipynb` fait tout dans Colab : chargement de Gemma 4 et de l'adaptateur LoRA, puis lancement de l'app (section 8). Dans Colab ouvert dans le navigateur, l'interface s'affiche sous la cellule. Depuis VS Code, l'extension Colab ne transfère pas les ports : l'interface Gradio ne peut pas s'y afficher.

## Direction visuelle

- Portrait animé dans `assets/animated.gif`, à côté du titre.
- Portrait fixe `assets/7.png` si la préférence système réduit les animations.
- Mise en page adaptée aux petits écrans et suggestions de questions en français.

Le style est défini dans `app/style.css` ; les composants et interactions Gradio dans `app/app.py`.
