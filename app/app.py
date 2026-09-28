import gradio as gr


# ============================================================
# CONFIGURATION
# ============================================================

MAX_HISTORY_TURNS = 5


SYSTEM_PROMPT = """
Tu es une simulation historique à vocation pédagogique,
inspirée du registre public de François Mitterrand autour de 1981.

Tu ne prétends pas être réellement François Mitterrand.
""".strip()


# ============================================================
# HISTORIQUE
# ============================================================

def limit_history(history):
    """
    Garde uniquement les derniers échanges.
    Cette fonction servira plus tard pour limiter
    le contexte envoyé au modèle.
    """

    if not history:
        return []

    return history[-MAX_HISTORY_TURNS * 2:]


# ============================================================
# AJOUT DU MESSAGE UTILISATEUR
# ============================================================

def add_user_message(message, history):
    """
    Ajoute le message de l'utilisateur dans le chat.
    """

    if history is None:
        history = []

    # Ignore les messages vides
    if not message or not message.strip():
        return "", history

    history = history.copy()

    history.append({
        "role": "user",
        "content": message.strip()
    })

    # Vide la zone de saisie
    return "", history


# ============================================================
# RÉPONSE TEMPORAIRE
# ============================================================

def generate_response(history):
    """
    Fonction temporaire.

    Pour l'instant, aucun modèle n'est chargé.
    Cette fonction sera remplacée plus tard
    par l'appel à Gemma + LoRA.
    """

    if not history:
        return history

    # Dernier message envoyé par l'utilisateur
    user_message = history[-1]["content"]

    # Réponse fictive pour tester l'interface
    answer = (
        "Mode démonstration — aucun modèle n'est encore chargé.\n\n"
        f"Vous avez demandé : « {user_message} »\n\n"
        "La réponse de Gemma fine-tuné apparaîtra ici lorsque "
        "le modèle et l'adaptateur LoRA seront intégrés."
    )

    history = history.copy()

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history


# ============================================================
# RESET
# ============================================================

def reset_chat():
    """
    Efface complètement la conversation.
    """

    return [], ""


# ============================================================
# INTERFACE
# ============================================================

with gr.Blocks(
    css_paths="style.css",
    title="Mitterrand 1981"
) as demo:

    # --------------------------------------------------------
    # TITRE
    # --------------------------------------------------------

    gr.Markdown(
        """
        # François Mitterrand — 1981
        ### Agent conversationnel historique
        """,
        elem_id="header"
    )


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    gr.Markdown(
        """
        Cet agent est une **simulation historique expérimentale**
        autour des positions et du registre politique de
        François Mitterrand en 1981.

        **Limites :**

        - Il ne s'agit pas réellement de François Mitterrand.
        - Le modèle est actuellement en cours d'entraînement.
        - Cette version utilise des réponses fictives pour tester l'interface.
        """,
        elem_id="information"
    )


    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    chatbot = gr.Chatbot(
        value=[],
        height=500,
        layout="bubble",
        elem_id="chatbot"
    )


    # --------------------------------------------------------
    # SAISIE
    # --------------------------------------------------------

    with gr.Row(elem_id="input-row"):

        message_input = gr.Textbox(
            placeholder="Posez votre question...",
            show_label=False,
            lines=2,
            max_lines=5,
            elem_id="message-input",
            scale=8
        )

        send_button = gr.Button(
            "Envoyer",
            variant="primary",
            elem_id="send-button",
            scale=1
        )


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    clear_button = gr.Button(
        "Réinitialiser la conversation",
        elem_id="clear-button"
    )


    # --------------------------------------------------------
    # EXEMPLES
    # --------------------------------------------------------

    gr.Examples(
        examples=[
            ["Quelle politique proposez-vous contre le chômage ?"],
            ["Pourquoi souhaitez-vous abolir la peine de mort ?"],
            ["Quelle est votre position sur les nationalisations ?"],
            ["Quelle conception avez-vous de l'Europe ?"],
        ],
        inputs=message_input
    )


    # ========================================================
    # ÉVÉNEMENTS
    # ========================================================

    # Bouton envoyer
    send_button.click(
        fn=add_user_message,
        inputs=[message_input, chatbot],
        outputs=[message_input, chatbot],
        queue=False
    ).then(
        fn=generate_response,
        inputs=chatbot,
        outputs=chatbot
    )


    # Touche Entrée
    message_input.submit(
        fn=add_user_message,
        inputs=[message_input, chatbot],
        outputs=[message_input, chatbot],
        queue=False
    ).then(
        fn=generate_response,
        inputs=chatbot,
        outputs=chatbot
    )


    # Bouton reset
    clear_button.click(
        fn=reset_chat,
        outputs=[chatbot, message_input],
        queue=False
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    demo.queue()
    demo.launch()