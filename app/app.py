from base64 import b64encode
from pathlib import Path

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR.parent / "assets"
MAX_HISTORY_TURNS = 5

# Fonction respond(question, history) -> str fournie par launch_app ; None = mode démonstration.
_respond = None

SYSTEM_PROMPT = """
Tu es une simulation historique à vocation pédagogique,
inspirée du registre public de François Mitterrand autour de 1981.
Tu ne prétends pas être réellement François Mitterrand.
""".strip()


def image_uri(name, mime):
    """Intègre les images locales, y compris depuis un notebook."""
    return f"data:{mime};base64," + b64encode((ASSETS_DIR / name).read_bytes()).decode("ascii")


def limit_history(history):
    """Limite le contexte qui sera envoyé au modèle."""
    return history[-MAX_HISTORY_TURNS * 2:] if history else []


def message_text(content):
    """Texte d'un message : Gradio 6 renvoie une liste de blocs, les messages ajoutés ici sont des chaînes."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if block.get("type") == "text")


def add_user_message(message, history):
    if not message or not message.strip():
        return "", history or []
    return "", [*(history or []), {"role": "user", "content": message.strip()}]


def generate_response(history):
    """Réponse du modèle branché par launch_app(respond=...), ou message de démonstration."""
    if not history or history[-1]["role"] != "user":
        return history
    question = message_text(history[-1]["content"])
    if _respond is None:
        answer = (
            "Mode démonstration — aucun modèle n'est encore chargé.\n\n"
            f"Vous avez demandé : « {question} »\n\n"
            "La réponse de la simulation historique apparaîtra ici lorsque "
            "le modèle sera intégré."
        )
    else:
        previous = [
            {"role": message["role"], "content": message_text(message["content"])}
            for message in limit_history(history[:-1])
        ]
        answer = _respond(question, previous)
    return [*history, {"role": "assistant", "content": answer}]


def reset_chat():
    return [], ""


# Les images sont intégrées au composant, sans serveur de fichiers supplémentaire.
with gr.Blocks(
    title="Mitterrand 1981 · Une conversation avec l’histoire",
    analytics_enabled=False,
) as demo:
    gr.HTML(
        f"""
        <header class="masthead" lang="fr">
          <div class="masthead-copy">
            <h1><span class="given-name">François</span> MITTERRAND <span class="year">1981</span></h1>
            <p class="tagline">Une conversation avec <span>l’histoire.</span></p>
            <p class="intro">Idées, débats et convictions : explorez la France de  François Mitterrand et ses 101 propositions.</p>
          </div>
          <figure class="portrait">
            <img class="portrait-animated" src="{image_uri('animated.gif', 'image/gif')}"
                 alt="Portrait animé de François Mitterrand, en noir et blanc et touches pop"
                 width="576" height="576">
            <img class="portrait-static" src="{image_uri('7.png', 'image/png')}"
                 alt="Portrait de François Mitterrand en collage pop" width="576" height="576">
            <figcaption>PORTRAIT D’UNE GAUCHE QUI GAGNE</figcaption>
          </figure>
        </header>
        """,
        elem_id="header",
        js_on_load=None,
    )
    gr.Markdown(
        "**SIMULATION HISTORIQUE** · Un dialogue inspiré du registre "
        "politique de François Mitterrand entre autour de 1981. <span>Ce projet est réalisé dans le cadre du cours de NLP à CY Tech</span>",
        elem_id="information",
    )
    with gr.Accordion("Le rôle et les limites de cet agent", open=False, elem_id="context"):
        gr.Markdown(
            "Cet agent permet de découvrir les positions et les débats politiques de 1981. "
            "Il ne s’agit pas réellement de François Mitterrand. Les réponses ne sont pas "
            "des citations historiques et doivent être confrontées à des sources.\n\n"
            "Le modèle est un LLM re-entrainé via LoRA sur une base de données de citations publiques pour coller au mieux au style et aux positions de François Mitterrand."
        )

    gr.HTML(
        '<div class="section-heading" lang="fr"><h2>À vous la parole.</h2>',
        js_on_load=None,
    )
    # Saisie et sujets au-dessus de la conversation : on écrit sans avoir à défiler sous le fil.
    with gr.Row(elem_id="input-row"):
        message_input = gr.Textbox(
            placeholder="Votre question pour François Mitterrand…",
            label="Votre message", show_label=False,
            lines=2, max_lines=5, elem_id="message-input", scale=8,
        )
        send_button = gr.Button(
            "Envoyer ↗", variant="primary", elem_id="send-button", scale=1, min_width=140,
        )

    gr.Markdown("POUR OUVRIR LE DÉBAT", elem_id="topics-label")
    topics = [
        ("Emploi", "Quelle politique proposez-vous contre le chômage ?"),
        ("Peine de mort", "Pourquoi souhaitez-vous abolir la peine de mort ?"),
        ("Nationalisations", "Quelle est votre position sur les nationalisations ?"),
        ("Europe", "Quelle conception avez-vous de l’Europe ?"),
    ]
    with gr.Row(elem_id="topics"):
        topic_buttons = [gr.Button(label, size="sm", min_width=120) for label, _ in topics]

    chatbot = gr.Chatbot(
        value=[], height=390, layout="bubble",
        label="Conversation avec Mitterrand — simulation",
        show_label=False, buttons=[], feedback_options=[],
        placeholder="<strong>Et si vous posiez la question ?</strong><br>"
                    "Économie, société, Europe… Choisissez un sujet ou écrivez le vôtre.",
        elem_id="chatbot",
    )

    with gr.Row(elem_id="conversation-tools"):
        clear_button = gr.Button(
            "Réinitialiser la conversation", elem_id="clear-button", size="sm", scale=0,
        )

    for button, (_, question) in zip(topic_buttons, topics):
        button.click(fn=lambda q=question: q, outputs=message_input, queue=False)

    gr.HTML(
        '<footer class="page-footer" lang="fr"><span>MITTERRAND · 1981</span>'
        '<span>Projet de NLP à CY Tech</span></footer>',
        js_on_load=None,
    )

    for event in (send_button.click, message_input.submit):
        event(
            fn=add_user_message, inputs=[message_input, chatbot],
            outputs=[message_input, chatbot], queue=False,
        ).then(fn=generate_response, inputs=chatbot, outputs=chatbot)

    clear_button.click(fn=reset_chat, outputs=[chatbot, message_input], queue=False)


def launch_app(respond=None, **kwargs):
    """Lance l'interface stylisée depuis un script ou un notebook (Gradio 6).

    respond(question, history) -> str interroge le modèle ; history contient les derniers
    échanges au format [{"role": ..., "content": str}]. Sans respond, l'app reste en démonstration.
    """
    global _respond
    _respond = respond
    kwargs.setdefault("css_paths", str(APP_DIR / "style.css"))
    kwargs.setdefault("footer_links", [])
    # Dans Colab, Gradio créerait sinon un lien public gradio.live : sans lui, l'app s'affiche sous la cellule.
    kwargs.setdefault("share", False)
    demo.queue()
    return demo.launch(**kwargs)


if __name__ == "__main__":
    launch_app()
