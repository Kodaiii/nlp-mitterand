# Dataset QLoRA « Mitterrand 1981 »

Paires user → assistant pour fine-tuner un modèle de chat (Gemma ou autre) à parler comme François Mitterrand candidat puis président élu au printemps 1981 : sa manière de parler, et le contenu des *110 propositions pour la France*.

## Fichiers

| Fichier | Contenu |
|---|---|
| `mitterrand_train.jsonl` | 490 exemples d'entraînement |
| `mitterrand_val.jsonl` | 58 exemples de validation (≈10 %, stratifiés par catégorie, seed 42) |
| `mitterrand_test.jsonl` | 30 exemples de test (≈5 %, stratifiés par catégorie, seed 42), réservé à l'évaluation finale |
| `mitterrand_full_with_meta.jsonl` | les 578 exemples avec `id`, `category` et `props` (numéros de propositions couverts) |
| `sources/*.md` | sources lisibles et éditables, d'où sont générés les JSONL |
| `system_prompt.txt` | prompt système placé en tête de chaque exemple |

Format (conversationnel TRL / Unsloth) :

```json
{"messages": [{"role": "system", "content": "Tu es François Mitterrand au printemps 1981…"}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Chaque exemple commence par le même prompt système (`system_prompt.txt`, 181 mots). Il décrit le personnage, le style (vouvoiement, prose sans mise en forme), l'ancrage au printemps 1981 et la conduite à tenir si on lui demande s'il est une IA. **À l'inférence, il faut réutiliser exactement ce prompt** : le modèle aura appris à y associer le personnage.

Gemma 4 a un tour système natif dans son chat template (`<|turn>system`). Gemma 2 et 3 n'ont pas de rôle système dédié : leur template le fusionne dans le premier message utilisateur, ou lève une erreur. Pour ces modèles-là, ou pour apprendre le personnage sans prompt, régénérez avec `--no-system-prompt`. Le chat template convertit `assistant` en `model`.

## Contenu

| Catégorie | Nb | Rôle |
|---|---|---|
| propositions | 110 | une question par proposition (1 à 110) |
| thematiques | 63 | questions transversales (emploi, fiscalité, Europe…) |
| contradicteurs | 54 | questions hostiles, pièges, débat Giscard |
| citoyens | 46 | sidérurgiste, agricultrice, étudiante, immigré… |
| citations | 91 | chaque citation du CSV reprise mot pour mot dans une réponse |
| biographie | 49 | Jarnac, captivité, Résistance, IVe République, Épinay, 10 mai 1981 |
| multitours | 35 | dialogues de 2 à 3 tours |
| divers_hors_epoque | 40 | politesses, réponses courtes, sujets postérieurs à 1981 |
| discours | 12 | allocutions longues (190 à 300 mots) |
| faits_courts | 55 | chiffres et engagements, dont « que dit la proposition n° X » |
| registres_familiers | 23 | tutoiement, fautes de frappe, messages très courts |

Toutes les propositions (110/110) et toutes les citations (91/91) sont couvertes. Les réponses font 106 mots en moyenne (entre 8 et 297 mots). Aucune ne contient de markdown, puisque le personnage parle en prose.

## Choix de conception

- **Ancrage temporel au printemps 1981.** Sur Internet, l'IA, le Covid ou Macron, le personnage dit qu'il ne connaît pas et raisonne à partir de ses principes de 1981.
- **Question « es-tu une IA ? »** (2 exemples). Il reconnaît être une voix reconstituée, puis propose de continuer le jeu. Retirez ces exemples de `08_divers_hors_epoque.md` si vous préférez qu'il reste entièrement dans le personnage.
- **Vouvoiement systématique**, même quand l'utilisateur le tutoie.
- **Tournures variées.** Les tics répétés ont été réduits, par exemple « Mon programme prévoit », alterné avec trois équivalents. Aucune citation n'apparaît plus de 4 fois.

## Points de vigilance

- **Numérotation du CSV.** Les questions « proposition n° X » suivent la numérotation de `110_propositions_mitterrand_1981.csv`. La seconde moitié de ce fichier (environ 61 à 110) ne correspond pas toujours au document historique. On y trouve aussi des anachronismes : « ORTF » pour la proposition 63 (l'ORTF a été dissous en 1974) et « CHSCT » pour la 77 (les CHSCT ont été créés en 1982). Les réponses parlent donc de « monopole d'État sur la radio et la télévision » et de « comités d'hygiène et de sécurité ». Il vaut mieux vérifier la numérotation avant d'évaluer le modèle dessus.
- **Citations.** Leur attribution n'a pas été vérifiée : elles sont reprises telles quelles depuis le CSV. Certaines datent d'après 1981, par exemple celle sur l'Allemagne ou celle sur l'entreprise ; elles sont utilisées pour le style.
- **Faits biographiques.** Seuls des faits établis ont été retenus. Les anecdotes dont la source était douteuse ont été retirées ou atténuées.

## Régénérer / modifier

Éditez `sources/*.md` (un exemple = un bloc `@@@`, balises `[USER]`, `[ASSISTANT]`, `[PROPS]` optionnelle), puis :

```bash
python3 scripts/build_dataset.py
```

Pour modifier le prompt système, éditez `system_prompt.txt` et relancez le script. Pour générer une version sans prompt système : `python3 scripts/build_dataset.py --no-system-prompt`.

Le script valide l'alternance des rôles, rejette les doublons et le markdown, puis affiche la couverture des propositions et des citations.

## Charger pour l'entraînement

Le notebook Colab [`agent.ipynb`](../agent.ipynb) regroupe le QLoRA sur `google/gemma-4-12B-it` (chargement 4 bits, masquage de la perte sur les seules r?ponses, entraînement, comparaison avant/après et sauvegarde) et l'interface Gradio. Avec `RUN_TRAINING = False`, il recharge directement l'adaptateur depuis Drive ou Hugging Face, sans réentraîner. Il peut être ouvert dans Colab ou depuis VS Code avec l'extension officielle Google Colab ; pour afficher Gradio, utiliser Colab dans le navigateur. Pour un autre outil :

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "data/mitterrand_train.jsonl", "validation": "data/mitterrand_val.jsonl", "test": "data/mitterrand_test.jsonl"})
```

Avec TRL `SFTTrainer`, le format `messages` est reconnu directement. Il vaut la peine de ne calculer la perte que sur les réponses du personnage (`assistant_only_loss=True`, qui demande un chat template avec les balises `{% generation %}`, ou l'équivalent `train_on_responses_only` d'Unsloth). Sinon, le modèle apprend aussi le style des questions.
