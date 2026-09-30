MITTERRAND 1981 — UNE CONVERSATION AVEC L’HISTOIRE

![Animation](assets/animated.gif)

Agent conversationnel historique en français réaliser dans le cadre du cours de NLP.
Point d’entrée : agent.ipynb.

STRUCTURE
agent.ipynb             Observation du modèle, baseline, QLoRA, évaluation et Gradio.
app/app.py             Composants et interactions de l’interface.
app/style.css          Style de l’interface.
assets/                Portraits fixes et animés.
data/                  Dialogues, splits, prompt et documentation détaillée.
scripts/build_dataset.py  Reconstruction reproductible des données.

EXÉCUTION DU NOTEBOOK
1. Ouvrir agent.ipynb dans Colab, avec Chrome ou Firefox.
2. Choisir un GPU L4 ou A100 pour l’entraînement.
3. Accepter la licence du modèle google/gemma-4-12B-it sur Hugging Face.
4. Ajouter le secret Colab HF_TOKEN, ou utiliser la connexion interactive.
5. Choisir le parcours dans la section Configuration :
   - RUN_TRAINING = True : entraîner, sauvegarder puis discuter.
   - RUN_TRAINING = False (par défaut) : charger l’adaptateur depuis Drive.
     Garder ADAPTER_PATH = "/content/drive/MyDrive/mitterrand-qlora".
     Placer les fichiers directement dans Mon Drive/mitterrand-qlora/ :
     adapter_config.json, adapter_model.safetensors et les fichiers du
     tokenizer/processor, dont chat_template.jinja. Ne pas ajouter de
     sous-dossier adapter. Drive est monté automatiquement en section 14.
     Autoriser la connexion à Drive lorsque Colab le demande.
6. Vérifier REPO_BRANCH : main.
7. Exécuter les cellules dans l’ordre. Le modèle de base est chargé en
   section 4, observé en section 5 et comparé à trois prompts en section 6
   (baseline). La section 15 évalue la baseline et le modèle adapté.
   L’interface Gradio s’affiche en section 16.

Les résultats de la baseline et de l’évaluation sont enregistrés dans
Mon Drive/mitterrand-eval/ (RESULTS_DIR), ou dans evaluation/ hors de Colab :
baseline_reference.json, reponses.json et grille.csv. Les réponses déjà
générées sont relues à la session suivante au lieu d’être régénérées.

Dans Colab, le code et les ressources du projet sont clonés automatiquement.
L’adaptateur est chargé séparément depuis Drive. En local,
ouvrir le notebook depuis le dépôt ; les données et l’app sont trouvées
sur place. Le chargement quantifié du modèle nécessite un GPU compatible.
L’extension Colab de VS Code ne transfère pas les ports Gradio : utiliser
Colab dans le navigateur pour la démonstration.

Le tokenizer et le chat template sauvegardés avec l’adaptateur sur Drive
sont réutilisés. Les nouveaux entraînements sont sauvegardés dans
Mon Drive/mitterrand-qlora/ lorsque SAVE_TO_DRIVE = True.
Le prompt système sauvegardé avec l’adaptateur est réutilisé. Pour un ancien
adaptateur sans prompt, data/system_prompt.txt sert de référence.
L’interface conserve les cinq derniers échanges comme contexte.

INTERFACE SEULE (DÉMONSTRATION SANS MODÈLE)
Depuis la racine du projet, dans un environnement contenant Gradio 6 :
    python app/app.py
Ou avec l’environnement virtuel local sous Windows :
    .venv\Scripts\python.exe app\app.py
Ouvrir l’adresse indiquée dans le terminal. Pour utiliser le modèle, passer
par agent.ipynb, qui lance launch_app(respond=ask).

DONNÉES
Le jeu courant comporte 578 exemples : train 490, validation 58, test 30.
Voir data/README.md pour les catégories, les sources et les limites.
Pour reconstruire les JSONL depuis data/sources/ :
    python scripts/build_dataset.py

REMISE
1. Exécuter agent.ipynb jusqu’à la section 15 : elle génère les réponses
   des deux modèles et crée grille.csv dans RESULTS_DIR.
2. Noter grille.csv à la main (0, 1 ou 2 par critère, NA déjà rempli quand
   le critère ne s’applique pas), l’enregistrer au même emplacement, puis
   relancer les cellules de synthèse de la section 15.
3. Rédiger au moins cinq réussites ou erreurs commentées (fin de la
   section 15), puis exécuter le notebook jusqu’à l’interface.
4. Joindre au rendu le contenu de RESULTS_DIR (dossier evaluation/).
Restent à préparer : rapport.pdf (4 pages maximum hors annexes) et le
lien stable vers l’adaptateur.
