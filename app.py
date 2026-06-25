from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from src.audio_cleaner import clean_audio_pipeline
from src.config import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_END_PADDING_SECONDS,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_POST_SPEECH_PADDING_MS,
    DEFAULT_PRE_SPEECH_PADDING_MS,
    DEFAULT_SHORTS_SUBTITLE_FONT_SIZE,
    DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD,
    DEFAULT_SHORTS_SUBTITLE_COLOR,
    DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR,
    DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE,
    DEFAULT_SHORTS_SUBTITLE_BASE_FONT,
    DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT,
    DEFAULT_SHORTS_SUBTITLE_MARGIN_V,
    DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE,
    DEFAULT_SHORTS_SUBTITLE_SPACING,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_START_PADDING_SECONDS,
    DEFAULT_WHISPER_MODEL,
    OUTPUTS_DIR,
    PROJECT_NAME,
    SHORTS_DIR,
    SHORTS_SUBTITLE_FONT_OPTIONS,
    UPLOADS_DIR,
    TRANSCRIPTS_DIR,
    SUBTITLED_VIDEOS_DIR,
    ensure_directories,
)
from src.transcriber import transcribe_media
from src.transcript_exporter import (
    format_timestamped_transcript,
    transcript_rows,
    write_timestamped_csv,
    write_timestamped_txt,
    write_transcript_json,
)
from src.utils import unique_path
from src.video_cutter import generate_short_from_text, generate_video_with_subtitles


ensure_directories()

st.set_page_config(page_title=PROJECT_NAME, layout="wide", initial_sidebar_state="expanded")

TRANSLATIONS = {
    "pt": {
        "app_tagline": "Assistente local para limpar audio e gerar Shorts a partir de videos longos.",
        "home_eyebrow": "Apresentacao inicial",
        "home_title": "Escolha o fluxo e siga sem precisar entender Streamlit.",
        "home_body": (
            "A pagina inicial apresenta as quatro rotas atuais do projeto. "
            "Use o menu lateral para entrar diretamente no fluxo desejado."
        ),
        "home_stat_one_label": "Fluxos atuais",
        "home_stat_one_value": "4",
        "home_stat_two_label": "Upload maximo",
        "home_stat_two_value": "1 GB",
        "home_stat_three_label": "Idioma da interface",
        "home_stat_three_value": "PT / EN",
        "clean_audio_title": "Limpar audio para DaVinci Resolve",
        "clean_audio_description": "Transcreve, remove silencias e gera entregaveis para edicao.",
        "transcription_title": "Transcrever video com timestamps",
        "transcription_description": "Gera a transcricao segmentada com tempos para revisao e copia.",
        "shorts_title": "Gerar Shorts",
        "shorts_description": "Encontra um trecho exato no texto e gera um video vertical.",
        "subtitle_burn_title": "Adicionar legenda no video",
        "subtitle_burn_description": "Queima a legenda no video com o estilo atual ou com um arquivo enviado.",
        "nav_clean_audio": "Limpar audio",
        "nav_transcription": "Transcricao com timestamps",
        "nav_shorts": "Gerar Shorts",
        "nav_subtitle_burn": "Adicionar legenda no video",
        "back_home": "Voltar a apresentacao",
        "sidebar_title": "Menu",
        "sidebar_caption": "Escolha uma das quatro rotas disponiveis.",
        "language_label": "Idioma / Language",
        "advanced_settings": "Configuracoes avancadas",
        "model_label": "Modelo Whisper",
        "device_label": "Dispositivo",
        "home_cta_clean": "Abrir limpeza de audio",
        "home_cta_transcription": "Abrir transcricao",
        "home_cta_shorts": "Abrir Gerador de Shorts",
        "home_cta_subtitle_burn": "Abrir legenda no video",
        "audio_header": "Limpeza de audio",
        "transcription_header": "Transcricao com timestamps",
        "video_header": "Geracao de Shorts",
        "subtitle_burn_header": "Adicionar legenda no video",
        "upload_audio": "Enviar audio",
        "upload_media": "Enviar video ou audio",
        "upload_video": "Enviar video",
        "upload_video_subtitle": "Enviar video para legenda",
        "upload_subtitle_file": "Enviar arquivo de legenda",
        "upload_subtitle_file_help": "Opcional. Se enviar um arquivo .ASS ou .SRT, ele sera usado no video.",
        "upload_audio_help": "Aceita arquivos de audio com ate 1 GB.",
        "upload_video_help": "Aceita arquivos de video com ate 1 GB.",
        "upload_media_help": "Aceita arquivos de video ou audio com ate 1 GB.",
        "script_text": "Texto do roteiro",
        "script_placeholder": "Cole aqui o roteiro completo.",
        "silence_threshold": "Limiar de silencio (dB)",
        "minimum_silence": "Silencio minimo (ms)",
        "pre_padding": "Margem antes da fala (ms)",
        "post_padding": "Margem depois da fala (ms)",
        "duplicate_similarity": "Similaridade para duplicacao",
        "process_audio": "Processar audio",
        "processing_audio_success": "Audio processado com sucesso.",
        "progress_audio_title": "Processando limpeza de audio...",
        "progress_audio_loading_model": "Carregando modelo Whisper",
        "progress_audio_transcribing": "Transcrevendo audio",
        "progress_audio_transcription_cached": "Usando transcricao em cache",
        "progress_audio_detecting_silences": "Detectando silencias",
        "progress_audio_detecting_duplicates": "Procurando trechos duplicados",
        "progress_audio_rendering_audio": "Montando audio limpo",
        "progress_audio_exporting_mp3": "Exportando MP3",
        "progress_audio_saving_reports": "Salvando relatorios",
        "progress_audio_done": "Limpeza concluida",
        "cuda_fallback_warning": "CUDA falhou neste ambiente; o processamento continuou em CPU.",
        "progress_transcription_title": "Gerando transcricao...",
        "progress_transcription_loading_model": "Carregando modelo Whisper",
        "progress_transcription_transcribing": "Transcrevendo midia",
        "progress_transcription_exporting": "Exportando arquivos",
        "progress_transcription_done": "Transcricao concluida",
        "progress_burn_title": "Adicionando legenda ao video...",
        "progress_burn_transcribing": "Transcrevendo video",
        "progress_burn_building_subtitles": "Gerando legenda estilizada",
        "progress_burn_rendering_video": "Renderizando video com legenda",
        "progress_burn_saving_metadata": "Salvando metadados",
        "progress_burn_done": "Video legendado concluido",
        "transcript_label": "Transcricao",
        "transcript_timestamps": "Transcricao com timestamps",
        "transcript_segments": "Segmentos",
        "downloads_label": "Downloads",
        "download_transcript_txt": "Baixar TXT com timestamps",
        "download_transcript_csv": "Baixar CSV",
        "download_transcript_json": "Baixar JSON",
        "download_cleaned_wav": "Baixar WAV limpo",
        "download_cleaned_mp3": "Baixar MP3 limpo",
        "download_report_json": "Baixar relatorio JSON",
        "download_report_csv": "Baixar relatorio CSV",
        "download_davinci_wav": "Baixar WAV para DaVinci",
        "download_davinci_csv": "Baixar CSV para DaVinci",
        "video_intro": "Adicione ate 2 Shorts.",
        "short_title": "Titulo",
        "short_description": "Descricao",
        "short_excerpt": "Trecho exato",
        "short_excerpt_placeholder": "Cole o trecho exato que deseja localizar na transcricao.",
        "start_padding": "Margem inicial (s)",
        "end_padding": "Margem final (s)",
        "zoom_out": "Zoom out do video",
        "zoom_out_help": "Aumente para abrir espaco apenas no topo e no rodape com fundo desfocado.",
        "subtitle_color": "Cor da legenda",
        "subtitle_highlight_color": "Cor da palavra ativa",
        "subtitle_font_size": "Tamanho da fonte",
        "subtitle_outline_size": "Espessura do contorno",
        "subtitle_shadow_size": "Sombra",
        "subtitle_margin_v": "Altura da legenda",
        "subtitle_spacing": "Espaçamento entre letras",
        "subtitle_base_font": "Fonte principal",
        "subtitle_accent_font": "Fonte de destaque",
        "subtitle_accent_last_word": "Destacar ultima palavra",
        "subtitle_style": "Estilo da legenda",
        "match_threshold": "Limite de correspondencia",
        "generate_shorts": "Gerar Shorts",
        "short_generated": "Short gerado com sucesso.",
        "progress_shorts_title": "Gerando Shorts...",
        "progress_shorts_loading_model": "Carregando modelo Whisper",
        "progress_shorts_transcribing": "Transcrevendo video",
        "progress_shorts_transcription_cached": "Usando transcricao em cache",
        "progress_shorts_matching_excerpt": "Localizando o trecho",
        "progress_shorts_building_subtitles": "Gerando legendas",
        "progress_shorts_rendering_video": "Renderizando video vertical",
        "progress_shorts_building_thumbnail": "Gerando miniatura",
        "progress_shorts_saving_metadata": "Salvando metadados",
        "progress_shorts_done": "Short concluido",
        "short_skipped_empty": "foi ignorado porque o trecho esta vazio.",
        "short_match_start_score": "Pontuacao de inicio",
        "short_match_end_score": "Pontuacao de fim",
        "short_start": "Inicio",
        "short_end": "Fim",
        "download_short_video": "Baixar MP4 do Short",
        "download_short_ass": "Baixar legenda ASS",
        "download_short_srt": "Baixar legenda SRT",
        "download_short_metadata": "Baixar metadados",
        "download_short_thumbnail": "Baixar thumbnail",
        "download_burned_video": "Baixar video com legenda",
        "download_used_subtitles": "Baixar legenda usada",
        "download_generated_ass": "Baixar legenda ASS gerada",
        "download_generated_srt": "Baixar legenda SRT gerada",
        "subtitle_source": "Fonte da legenda",
        "subtitle_source_auto": "Gerada automaticamente",
        "subtitle_source_uploaded": "Arquivo enviado",
        "subtitle_optional_note": "Se voce enviar um arquivo de legenda, ele tera prioridade. Se nao enviar, o app gera automaticamente com o estilo atual.",
        "transcribe_media": "Transcrever midia",
        "burn_subtitles": "Aplicar legenda",
        "davinci_ready": "Pronto para o DaVinci Resolve",
        "shorts_two_at_once": "Ate 2 Shorts por vez",
        "vertical_with_subtitles": "Formato vertical com legendas",
        "transcription_card_hint": "Transcricao pronta para revisão e copia.",
        "subtitle_burn_card_hint": "Legenda pronta com um clique, com opcao de arquivo externo.",
    },
    "en": {
        "app_tagline": "Local assistant for cleaning audio and generating Shorts from long videos.",
        "home_eyebrow": "Home presentation",
        "home_title": "Pick a workflow and get started without needing to learn Streamlit.",
        "home_body": (
            "The home screen introduces the four current routes in the project. "
            "Use the sidebar to jump straight to the workflow you want."
        ),
        "home_stat_one_label": "Current flows",
        "home_stat_one_value": "4",
        "home_stat_two_label": "Max upload",
        "home_stat_two_value": "1 GB",
        "home_stat_three_label": "Interface language",
        "home_stat_three_value": "PT / EN",
        "clean_audio_title": "Clean audio for DaVinci Resolve",
        "clean_audio_description": "Transcribes, removes silences, and builds editing deliverables.",
        "transcription_title": "Transcribe video with timestamps",
        "transcription_description": "Creates a segmented transcript with times for review and copying.",
        "shorts_title": "Generate Shorts",
        "shorts_description": "Finds an exact excerpt in the transcript and creates a vertical video.",
        "subtitle_burn_title": "Add subtitles to video",
        "subtitle_burn_description": "Burns subtitles into the video using the current style or an uploaded file.",
        "nav_clean_audio": "Clean audio",
        "nav_transcription": "Transcript with timestamps",
        "nav_shorts": "Generate Shorts",
        "nav_subtitle_burn": "Add subtitles to video",
        "back_home": "Back to presentation",
        "sidebar_title": "Menu",
        "sidebar_caption": "Choose one of the four available routes.",
        "language_label": "Idioma / Language",
        "advanced_settings": "Advanced settings",
        "model_label": "Whisper model",
        "device_label": "Device",
        "home_cta_clean": "Open audio cleanup",
        "home_cta_transcription": "Open transcription",
        "home_cta_shorts": "Open Shorts generator",
        "home_cta_subtitle_burn": "Open subtitle burner",
        "audio_header": "Audio cleanup",
        "transcription_header": "Transcript with timestamps",
        "video_header": "Shorts generation",
        "subtitle_burn_header": "Add subtitles to video",
        "upload_audio": "Upload audio",
        "upload_media": "Upload video or audio",
        "upload_video": "Upload video",
        "upload_video_subtitle": "Upload video to subtitle",
        "upload_subtitle_file": "Upload subtitle file",
        "upload_subtitle_file_help": "Optional. If you upload an .ASS or .SRT file, it will be used in the video.",
        "upload_audio_help": "Accepts audio files up to 1 GB.",
        "upload_video_help": "Accepts video files up to 1 GB.",
        "upload_media_help": "Accepts video or audio files up to 1 GB.",
        "script_text": "Script text",
        "script_placeholder": "Paste the full script here.",
        "silence_threshold": "Silence threshold (dB)",
        "minimum_silence": "Minimum silence (ms)",
        "pre_padding": "Pre-speech padding (ms)",
        "post_padding": "Post-speech padding (ms)",
        "duplicate_similarity": "Duplicate similarity",
        "process_audio": "Process audio",
        "processing_audio_success": "Audio processed successfully.",
        "progress_audio_title": "Processing audio cleanup...",
        "progress_audio_loading_model": "Loading Whisper model",
        "progress_audio_transcribing": "Transcribing audio",
        "progress_audio_transcription_cached": "Using cached transcription",
        "progress_audio_detecting_silences": "Detecting silences",
        "progress_audio_detecting_duplicates": "Looking for duplicate takes",
        "progress_audio_rendering_audio": "Building cleaned audio",
        "progress_audio_exporting_mp3": "Exporting MP3",
        "progress_audio_saving_reports": "Saving reports",
        "progress_audio_done": "Cleanup finished",
        "cuda_fallback_warning": "CUDA failed in this environment; processing continued on CPU.",
        "progress_transcription_title": "Generating transcript...",
        "progress_transcription_loading_model": "Loading Whisper model",
        "progress_transcription_transcribing": "Transcribing media",
        "progress_transcription_exporting": "Exporting files",
        "progress_transcription_done": "Transcript finished",
        "progress_burn_title": "Adding subtitles to video...",
        "progress_burn_transcribing": "Transcribing video",
        "progress_burn_building_subtitles": "Generating styled subtitles",
        "progress_burn_rendering_video": "Rendering video with subtitles",
        "progress_burn_saving_metadata": "Saving metadata",
        "progress_burn_done": "Subtitled video finished",
        "transcript_label": "Transcript",
        "transcript_timestamps": "Transcript with timestamps",
        "transcript_segments": "Segments",
        "downloads_label": "Downloads",
        "download_transcript_txt": "Download timestamped TXT",
        "download_transcript_csv": "Download CSV",
        "download_transcript_json": "Download JSON",
        "download_cleaned_wav": "Download cleaned WAV",
        "download_cleaned_mp3": "Download cleaned MP3",
        "download_report_json": "Download JSON report",
        "download_report_csv": "Download CSV report",
        "download_davinci_wav": "Download DaVinci WAV",
        "download_davinci_csv": "Download DaVinci CSV",
        "video_intro": "Add up to 2 Shorts.",
        "short_title": "Title",
        "short_description": "Description",
        "short_excerpt": "Exact excerpt",
        "short_excerpt_placeholder": "Paste the exact excerpt to find in the transcript.",
        "start_padding": "Start padding (s)",
        "end_padding": "End padding (s)",
        "zoom_out": "Video zoom out",
        "zoom_out_help": "Increase it to open space only at the top and bottom with a blurred background.",
        "subtitle_color": "Subtitle color",
        "subtitle_highlight_color": "Active word color",
        "subtitle_font_size": "Font size",
        "subtitle_outline_size": "Outline thickness",
        "subtitle_shadow_size": "Shadow",
        "subtitle_margin_v": "Subtitle height",
        "subtitle_spacing": "Letter spacing",
        "subtitle_base_font": "Base font",
        "subtitle_accent_font": "Accent font",
        "subtitle_accent_last_word": "Highlight last word",
        "subtitle_style": "Subtitle style",
        "match_threshold": "Match threshold",
        "generate_shorts": "Generate Shorts",
        "short_generated": "Short generated successfully.",
        "progress_shorts_title": "Generating Shorts...",
        "progress_shorts_loading_model": "Loading Whisper model",
        "progress_shorts_transcribing": "Transcribing video",
        "progress_shorts_transcription_cached": "Using cached transcription",
        "progress_shorts_matching_excerpt": "Locating the excerpt",
        "progress_shorts_building_subtitles": "Building subtitles",
        "progress_shorts_rendering_video": "Rendering vertical video",
        "progress_shorts_building_thumbnail": "Building thumbnail",
        "progress_shorts_saving_metadata": "Saving metadata",
        "progress_shorts_done": "Short finished",
        "short_skipped_empty": "was skipped because the excerpt is empty.",
        "short_match_start_score": "Start score",
        "short_match_end_score": "End score",
        "short_start": "Start",
        "short_end": "End",
        "download_short_video": "Download Short MP4",
        "download_short_ass": "Download ASS subtitles",
        "download_short_srt": "Download SRT subtitles",
        "download_short_metadata": "Download metadata",
        "download_short_thumbnail": "Download thumbnail",
        "download_burned_video": "Download subtitled video",
        "download_used_subtitles": "Download used subtitles",
        "download_generated_ass": "Download generated ASS",
        "download_generated_srt": "Download generated SRT",
        "subtitle_source": "Subtitle source",
        "subtitle_source_auto": "Auto-generated",
        "subtitle_source_uploaded": "Uploaded file",
        "subtitle_optional_note": "If you upload a subtitle file, it wins. Otherwise the app generates one automatically with the current style.",
        "transcribe_media": "Transcribe media",
        "burn_subtitles": "Apply subtitles",
        "davinci_ready": "Ready for DaVinci Resolve",
        "shorts_two_at_once": "Up to 2 Shorts at once",
        "vertical_with_subtitles": "Vertical format with subtitles",
        "transcription_card_hint": "Ready for review and copy.",
        "subtitle_burn_card_hint": "Burn subtitles with one click, or use your own file.",
    },
}

LANGUAGE_NAMES = {"pt": "Português", "en": "English"}


def t(key: str) -> str:
    language = st.session_state.get("ui_language", "pt")
    return TRANSLATIONS.get(language, TRANSLATIONS["pt"])[key]


def _set_view(view: str) -> None:
    current_view = st.session_state.get("view")
    if current_view != view:
        st.session_state.pop("audio_last_result", None)
        st.session_state.pop("shorts_last_results", None)
        st.session_state.pop("transcription_last_result", None)
        st.session_state.pop("subtitle_burn_last_result", None)
    st.session_state.view = view


def _rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if rerun is None:
        rerun = st.experimental_rerun
    rerun()


def _save_upload(uploaded_file, folder: Path) -> Path:
    source_path = Path(uploaded_file.name)
    target = unique_path(folder, source_path.stem, source_path.suffix or ".bin")
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _download_button(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True)


def _render_audio_result(result: dict[str, object]) -> None:
    st.subheader(t("transcript_label"))
    transcript_display = dict(result["transcript"])
    transcript_display["segments"] = [asdict(segment) for segment in result["transcript"]["segments"]]
    st.json(transcript_display, expanded=False)
    st.subheader(t("downloads_label"))
    _download_button(result["cleaned_wav"], t("download_cleaned_wav"), "audio/wav")
    _download_button(result["cleaned_mp3"], t("download_cleaned_mp3"), "audio/mpeg")
    _download_button(result["report_json"], t("download_report_json"), "application/json")
    _download_button(result["report_csv"], t("download_report_csv"), "text/csv")
    _download_button(result["davinci_wav"], t("download_davinci_wav"), "audio/wav")
    _download_button(result["davinci_csv"], t("download_davinci_csv"), "text/csv")


def _render_transcription_result(result: dict[str, object]) -> None:
    st.subheader(t("transcript_timestamps"))
    st.text_area(
        t("transcript_timestamps"),
        value=result["formatted_text"],
        height=320,
    )
    st.subheader(t("transcript_segments"))
    st.dataframe(result["rows"], use_container_width=True, hide_index=True)
    st.subheader(t("downloads_label"))
    _download_button(result["txt"], t("download_transcript_txt"), "text/plain")
    _download_button(result["csv"], t("download_transcript_csv"), "text/csv")
    _download_button(result["json"], t("download_transcript_json"), "application/json")


def _render_subtitle_burn_result(result: dict[str, object]) -> None:
    payload = result["payload"]
    st.subheader(t("subtitle_burn_header"))
    st.write(
        {
            t("subtitle_source"): t("subtitle_source_auto") if payload.get("subtitle_source") == "auto" else t("subtitle_source_uploaded"),
            t("subtitle_color"): payload.get("subtitle_color", DEFAULT_SHORTS_SUBTITLE_COLOR),
            t("subtitle_highlight_color"): payload.get("subtitle_highlight_color", DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR),
            t("subtitle_font_size"): payload.get("subtitle_font_size", DEFAULT_SHORTS_SUBTITLE_FONT_SIZE),
            t("subtitle_outline_size"): payload.get("subtitle_outline_size", DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE),
            t("subtitle_shadow_size"): payload.get("subtitle_shadow_size", DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE),
            t("subtitle_margin_v"): payload.get("subtitle_margin_v", DEFAULT_SHORTS_SUBTITLE_MARGIN_V),
            t("subtitle_spacing"): payload.get("subtitle_spacing", DEFAULT_SHORTS_SUBTITLE_SPACING),
            t("subtitle_base_font"): payload.get("subtitle_base_font", DEFAULT_SHORTS_SUBTITLE_BASE_FONT),
            t("subtitle_accent_font"): payload.get("subtitle_accent_font", DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT),
            t("subtitle_accent_last_word"): payload.get("subtitle_accent_last_word", DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD),
        }
    )
    st.subheader(t("downloads_label"))
    _download_button(result["video"], t("download_burned_video"), "video/mp4")
    if payload.get("subtitle_source") == "auto":
        if result.get("ass"):
            _download_button(result["ass"], t("download_generated_ass"), "text/plain")
        if result.get("srt"):
            _download_button(result["srt"], t("download_generated_srt"), "text/plain")
    elif result.get("subtitle_file"):
        _download_button(result["subtitle_file"], t("download_used_subtitles"), "text/plain")


def _render_subtitle_style_controls(key_prefix: str = "shorts") -> dict[str, object]:
    with st.expander(t("subtitle_style"), expanded=False):
        style_cols = st.columns(2)
        with style_cols[0]:
            subtitle_color = st.color_picker(
                t("subtitle_color"),
                DEFAULT_SHORTS_SUBTITLE_COLOR,
                key=f"{key_prefix}_subtitle_color",
            )
            subtitle_highlight_color = st.color_picker(
                t("subtitle_highlight_color"),
                DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR,
                key=f"{key_prefix}_subtitle_highlight_color",
            )
            subtitle_base_font = st.selectbox(
                t("subtitle_base_font"),
                options=SHORTS_SUBTITLE_FONT_OPTIONS,
                index=SHORTS_SUBTITLE_FONT_OPTIONS.index(DEFAULT_SHORTS_SUBTITLE_BASE_FONT),
                key=f"{key_prefix}_subtitle_base_font",
            )
        with style_cols[1]:
            subtitle_accent_font = st.selectbox(
                t("subtitle_accent_font"),
                options=SHORTS_SUBTITLE_FONT_OPTIONS,
                index=SHORTS_SUBTITLE_FONT_OPTIONS.index(DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT),
                key=f"{key_prefix}_subtitle_accent_font",
            )
            subtitle_accent_last_word = st.checkbox(
                t("subtitle_accent_last_word"),
                value=DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD,
                key=f"{key_prefix}_subtitle_accent_last_word",
            )
            subtitle_font_size = st.slider(
                t("subtitle_font_size"),
                56,
                110,
                DEFAULT_SHORTS_SUBTITLE_FONT_SIZE,
                step=1,
                key=f"{key_prefix}_subtitle_font_size",
            )
            subtitle_outline_size = st.slider(
                t("subtitle_outline_size"),
                0,
                10,
                DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE,
                step=1,
                key=f"{key_prefix}_subtitle_outline_size",
            )
        tail_cols = st.columns(2)
        with tail_cols[0]:
            subtitle_shadow_size = st.slider(
                t("subtitle_shadow_size"),
                0,
                8,
                DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE,
                step=1,
                key=f"{key_prefix}_subtitle_shadow_size",
            )
        with tail_cols[1]:
            subtitle_margin_v = st.slider(
                t("subtitle_margin_v"),
                80,
                420,
                DEFAULT_SHORTS_SUBTITLE_MARGIN_V,
                step=5,
                key=f"{key_prefix}_subtitle_margin_v",
            )
        subtitle_spacing = st.slider(
            t("subtitle_spacing"),
            0.0,
            3.0,
            DEFAULT_SHORTS_SUBTITLE_SPACING,
            step=0.1,
            key=f"{key_prefix}_subtitle_spacing",
        )
    return {
        "subtitle_color": subtitle_color,
        "subtitle_highlight_color": subtitle_highlight_color,
        "subtitle_font_size": subtitle_font_size,
        "subtitle_outline_size": subtitle_outline_size,
        "subtitle_shadow_size": subtitle_shadow_size,
        "subtitle_margin_v": subtitle_margin_v,
        "subtitle_spacing": subtitle_spacing,
        "subtitle_base_font": subtitle_base_font,
        "subtitle_accent_font": subtitle_accent_font,
        "subtitle_accent_last_word": subtitle_accent_last_word,
    }


def _render_shorts_results(results: list[dict[str, object]]) -> None:
    if not results:
        return
    st.subheader(t("downloads_label"))
    for index, result in enumerate(results, start=1):
        payload = result["payload"]
        display_title = payload["title"] or f"{t('short_title')} {index}"
        st.markdown(f"**{display_title}**")
        st.write(
            {
                t("short_match_start_score"): payload["match_start_score"],
                t("short_match_end_score"): payload["match_end_score"],
                t("short_start"): payload["start"],
                t("short_end"): payload["end"],
                t("subtitle_color"): payload.get("subtitle_color", DEFAULT_SHORTS_SUBTITLE_COLOR),
                t("subtitle_highlight_color"): payload.get("subtitle_highlight_color", DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR),
                t("subtitle_font_size"): payload.get("subtitle_font_size", DEFAULT_SHORTS_SUBTITLE_FONT_SIZE),
                t("subtitle_outline_size"): payload.get("subtitle_outline_size", DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE),
                t("subtitle_shadow_size"): payload.get("subtitle_shadow_size", DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE),
                t("subtitle_margin_v"): payload.get("subtitle_margin_v", DEFAULT_SHORTS_SUBTITLE_MARGIN_V),
                t("subtitle_spacing"): payload.get("subtitle_spacing", DEFAULT_SHORTS_SUBTITLE_SPACING),
                t("subtitle_base_font"): payload.get("subtitle_base_font", DEFAULT_SHORTS_SUBTITLE_BASE_FONT),
                t("subtitle_accent_font"): payload.get("subtitle_accent_font", DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT),
                t("subtitle_accent_last_word"): payload.get("subtitle_accent_last_word", DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD),
                t("zoom_out"): payload.get("zoom_out", 0.0),
            }
        )
        _download_button(result["video"], f"{t('download_short_video')} {index}", "video/mp4")
        _download_button(result["ass"], f"{t('download_short_ass')} {index}", "text/plain")
        _download_button(result["srt"], f"{t('download_short_srt')} {index}", "text/plain")
        _download_button(result["metadata"], f"{t('download_short_metadata')} {index}", "application/json")
        _download_button(result["thumbnail"], f"{t('download_short_thumbnail')} {index}", "image/jpeg")


def _make_progress_updater(progress_bar, status_placeholder, messages: dict[str, str]):
    def _update(stage: str, step: int, total: int) -> None:
        percent = 0 if total <= 0 else int(round((step / total) * 100))
        progress_bar.progress(min(100, max(0, percent)))
        status_placeholder.info(messages.get(stage, stage.replace("_", " ").title()))

    return _update


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .storycut-hero {
            padding: 2rem;
            border-radius: 1.25rem;
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.24), transparent 32%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96));
            color: white;
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 60px rgba(15, 23, 42, 0.18);
        }
        .storycut-hero h1 {
            margin: 0.2rem 0 0.6rem 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
        }
        .storycut-hero p {
            margin-bottom: 0;
            font-size: 1.02rem;
            line-height: 1.6;
            color: rgba(226, 232, 240, 0.95);
        }
        .storycut-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            color: rgba(191, 219, 254, 0.9);
            margin-bottom: 0.35rem;
        }
        .storycut-card {
            padding: 1.25rem;
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(248, 250, 252, 0.78);
            backdrop-filter: blur(6px);
            min-height: 12rem;
        }
        .storycut-card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .storycut-card p {
            margin-bottom: 0.75rem;
            color: #334155;
        }
        .storycut-card ul {
            margin: 0 0 1rem 1.1rem;
            color: #475569;
        }
        div.stButton > button[kind="primary"] {
            margin-top: 0.85rem;
            border: 1px solid rgba(96, 165, 250, 0.35);
            border-radius: 0.9rem;
            background: linear-gradient(135deg, #2563eb, #4f8cff);
            color: white;
            transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8, #3b82f6);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    st.sidebar.markdown(f"## {PROJECT_NAME}")
    st.sidebar.caption(t("app_tagline"))
    st.sidebar.caption(t("sidebar_caption"))
    st.sidebar.markdown(f"### {t('sidebar_title')}")
    st.sidebar.selectbox(
        t("language_label"),
        options=["pt", "en"],
        key="ui_language",
        format_func=lambda code: LANGUAGE_NAMES[code],
    )
    st.sidebar.markdown("---")
    if st.sidebar.button(t("nav_clean_audio"), use_container_width=True):
        _set_view("clean_audio")
    if st.sidebar.button(t("nav_transcription"), use_container_width=True):
        _set_view("transcription")
    if st.sidebar.button(t("nav_shorts"), use_container_width=True):
        _set_view("shorts")
    if st.sidebar.button(t("nav_subtitle_burn"), use_container_width=True):
        _set_view("subtitle_burn")
    st.sidebar.markdown("---")
    with st.sidebar.expander(t("advanced_settings"), expanded=False):
        st.selectbox(t("model_label"), ["small", "medium", "large-v3"], index=1, key="model_name")
        st.selectbox(t("device_label"), ["cpu", "cuda"], index=1, key="device")


def _render_home() -> None:
    st.markdown(
        f"""
        <div class="storycut-hero">
            <div class="storycut-eyebrow">{t("home_eyebrow")}</div>
            <h1>{PROJECT_NAME}</h1>
            <p>{t("home_title")}</p>
            <p>{t("home_body")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(3)
    metrics[0].metric(t("home_stat_one_label"), t("home_stat_one_value"))
    metrics[1].metric(t("home_stat_two_label"), t("home_stat_two_value"))
    metrics[2].metric(t("home_stat_three_label"), t("home_stat_three_value"))

    st.write("")
    first_row = st.columns(2)
    with first_row[0]:
        st.markdown(
            f"""
            <div class="storycut-card">
                <h3>{t("clean_audio_title")}</h3>
                <p>{t("clean_audio_description")}</p>
                <ul>
                    <li>{t("upload_audio_help")}</li>
                    <li>{t("downloads_label")}: WAV, MP3, JSON, CSV</li>
                    <li>{t("davinci_ready")}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        if st.button(t("home_cta_clean"), type="primary", use_container_width=True):
            _set_view("clean_audio")
            _rerun()
    with first_row[1]:
        st.markdown(
            f"""
            <div class="storycut-card">
                <h3>{t("transcription_title")}</h3>
                <p>{t("transcription_description")}</p>
                <ul>
                    <li>{t("upload_media_help")}</li>
                    <li>{t("transcription_card_hint")}</li>
                    <li>{t("download_transcript_txt")}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        if st.button(t("home_cta_transcription"), type="primary", use_container_width=True):
            _set_view("transcription")
            _rerun()

    second_row = st.columns(2)
    with second_row[0]:
        st.markdown(
            f"""
            <div class="storycut-card">
                <h3>{t("shorts_title")}</h3>
                <p>{t("shorts_description")}</p>
                <ul>
                    <li>{t("upload_video_help")}</li>
                    <li>{t("shorts_two_at_once")}</li>
                    <li>{t("vertical_with_subtitles")}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        if st.button(t("home_cta_shorts"), type="primary", use_container_width=True):
            _set_view("shorts")
            _rerun()
    with second_row[1]:
        st.markdown(
            f"""
            <div class="storycut-card">
                <h3>{t("subtitle_burn_title")}</h3>
                <p>{t("subtitle_burn_description")}</p>
                <ul>
                    <li>{t("upload_video_help")}</li>
                    <li>{t("subtitle_optional_note")}</li>
                    <li>{t("subtitle_burn_card_hint")}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        if st.button(t("home_cta_subtitle_burn"), type="primary", use_container_width=True):
            _set_view("subtitle_burn")
            _rerun()


def _render_clean_audio() -> None:
    st.subheader(t("audio_header"))
    if st.button(t("back_home")):
        _set_view("home")
        _rerun()

    uploaded_audio = st.file_uploader(t("upload_audio"), type=["wav", "mp3", "m4a", "aac"], help=t("upload_audio_help"))
    script_text = st.text_area(t("script_text"), height=220, placeholder=t("script_placeholder"))

    col1, col2 = st.columns(2)
    with col1:
        silence_threshold_db = st.slider(t("silence_threshold"), -60, -20, DEFAULT_SILENCE_THRESHOLD_DB)
        min_silence_duration_ms = st.slider(
            t("minimum_silence"),
            200,
            2000,
            DEFAULT_MIN_SILENCE_DURATION_MS,
            step=50,
        )
    with col2:
        pre_padding_ms = st.slider(t("pre_padding"), 0, 500, DEFAULT_PRE_SPEECH_PADDING_MS, step=10)
        post_padding_ms = st.slider(t("post_padding"), 0, 700, DEFAULT_POST_SPEECH_PADDING_MS, step=10)

    duplicate_threshold = st.slider(t("duplicate_similarity"), 50, 100, DEFAULT_DUPLICATE_THRESHOLD)

    if st.button(t("process_audio"), type="primary", disabled=uploaded_audio is None):
        st.session_state.pop("audio_last_result", None)
        progress_bar = None
        progress_status = None
        try:
            progress_bar = st.progress(0)
            progress_status = st.empty()
            progress_status.info(t("progress_audio_title"))
            progress_messages = {
                "loading_model": t("progress_audio_loading_model"),
                "transcribing": t("progress_audio_transcribing"),
                "transcription_cached": t("progress_audio_transcription_cached"),
                "transcription_done": t("progress_audio_transcribing"),
                "detecting_silences": t("progress_audio_detecting_silences"),
                "detecting_duplicates": t("progress_audio_detecting_duplicates"),
                "rendering_audio": t("progress_audio_rendering_audio"),
                "exporting_mp3": t("progress_audio_exporting_mp3"),
                "saving_reports": t("progress_audio_saving_reports"),
            }
            progress_callback = _make_progress_updater(progress_bar, progress_status, progress_messages)
            input_path = _save_upload(uploaded_audio, UPLOADS_DIR)
            result = clean_audio_pipeline(
                input_path,
                script_text=script_text,
                output_root=OUTPUTS_DIR,
                model_name=st.session_state.get("model_name", DEFAULT_WHISPER_MODEL),
                device=st.session_state.get("device", "cuda"),
                silence_threshold_db=silence_threshold_db,
                min_silence_duration_ms=min_silence_duration_ms,
                pre_speech_padding_ms=pre_padding_ms,
                post_speech_padding_ms=post_padding_ms,
                duplicate_threshold=duplicate_threshold,
                progress_callback=progress_callback,
            )
            progress_bar.progress(100)
            progress_status.success(t("progress_audio_done"))
            if not result["transcript"].get("cache_hit") and result["transcript"].get("device") != st.session_state.get("device", "cuda"):
                st.warning(t("cuda_fallback_warning"))
            st.session_state.audio_last_result = result
            st.success(t("processing_audio_success"))
        except Exception as exc:  # pragma: no cover - UI feedback
            if progress_status is not None:
                progress_status.error(str(exc))
            st.error(str(exc))

    audio_result = st.session_state.get("audio_last_result")
    if audio_result:
        _render_audio_result(audio_result)


def _render_transcription() -> None:
    st.subheader(t("transcription_header"))
    if st.button(t("back_home")):
        _set_view("home")
        _rerun()

    uploaded_media = st.file_uploader(
        t("upload_media"),
        type=["mp4", "mov", "mkv", "wav", "mp3", "m4a", "aac"],
        help=t("upload_media_help"),
    )

    if st.button(t("transcribe_media"), type="primary", disabled=uploaded_media is None):
        st.session_state.transcription_last_result = None
        progress_bar = None
        progress_status = None
        try:
            progress_bar = st.progress(0)
            progress_status = st.empty()
            progress_status.info(t("progress_transcription_title"))
            progress_messages = {
                "loading_model": t("progress_transcription_loading_model"),
                "transcribing": t("progress_transcription_transcribing"),
                "transcription_cached": t("progress_audio_transcription_cached"),
                "transcription_done": t("progress_transcription_exporting"),
            }
            progress_callback = _make_progress_updater(progress_bar, progress_status, progress_messages)
            input_path = _save_upload(uploaded_media, UPLOADS_DIR)
            transcript = transcribe_media(
                input_path,
                model_name=st.session_state.get("model_name", DEFAULT_WHISPER_MODEL),
                device=st.session_state.get("device", "cuda"),
                progress_callback=progress_callback,
            )
            segments = transcript["segments"]
            export_stem = unique_path(TRANSCRIPTS_DIR, input_path.stem, ".txt").stem
            txt_path = TRANSCRIPTS_DIR / f"{export_stem}.txt"
            csv_path = TRANSCRIPTS_DIR / f"{export_stem}.csv"
            json_path = TRANSCRIPTS_DIR / f"{export_stem}.json"
            formatted_text = format_timestamped_transcript(segments)
            write_timestamped_txt(txt_path, segments)
            write_timestamped_csv(csv_path, segments)
            write_transcript_json(json_path, transcript)
            result = {
                "transcript": transcript,
                "rows": transcript_rows(segments),
                "formatted_text": formatted_text,
                "txt": txt_path,
                "csv": csv_path,
                "json": json_path,
            }
            st.session_state.transcription_last_result = result
            if transcript.get("device") != st.session_state.get("device", "cuda"):
                st.warning(t("cuda_fallback_warning"))
            progress_bar.progress(100)
            progress_status.success(t("progress_transcription_done"))
        except Exception as exc:  # pragma: no cover - UI feedback
            if progress_status is not None:
                progress_status.error(str(exc))
            st.error(str(exc))

    transcription_result = st.session_state.get("transcription_last_result")
    if transcription_result:
        _render_transcription_result(transcription_result)


def _render_subtitle_burn() -> None:
    st.subheader(t("subtitle_burn_header"))
    if st.button(t("back_home")):
        _set_view("home")
        _rerun()

    uploaded_video = st.file_uploader(
        t("upload_video_subtitle"),
        type=["mp4", "mov", "mkv"],
        help=t("upload_video_help"),
    )
    uploaded_subtitle = st.file_uploader(
        t("upload_subtitle_file"),
        type=["ass", "srt"],
        help=t("upload_subtitle_file_help"),
    )
    st.caption(t("subtitle_optional_note"))
    subtitle_style = _render_subtitle_style_controls()

    if st.button(t("burn_subtitles"), type="primary", disabled=uploaded_video is None):
        st.session_state.subtitle_burn_last_result = None
        progress_bar = None
        progress_status = None
        try:
            progress_bar = st.progress(0)
            progress_status = st.empty()
            progress_status.info(t("progress_burn_title"))
            progress_messages = {
                "loading_model": t("progress_transcription_loading_model"),
                "transcribing": t("progress_burn_transcribing"),
                "transcription_cached": t("progress_audio_transcription_cached"),
                "transcription_done": t("progress_burn_building_subtitles"),
                "building_subtitles": t("progress_burn_building_subtitles"),
                "rendering_video": t("progress_burn_rendering_video"),
                "saving_metadata": t("progress_burn_saving_metadata"),
            }
            progress_callback = _make_progress_updater(progress_bar, progress_status, progress_messages)
            input_path = _save_upload(uploaded_video, UPLOADS_DIR)
            subtitle_path = _save_upload(uploaded_subtitle, UPLOADS_DIR) if uploaded_subtitle is not None else None
            result = generate_video_with_subtitles(
                input_path,
                output_dir=SUBTITLED_VIDEOS_DIR,
                model_name=st.session_state.get("model_name", DEFAULT_WHISPER_MODEL),
                device=st.session_state.get("device", "cuda"),
                subtitle_path=subtitle_path,
                subtitle_color=subtitle_style["subtitle_color"],
                subtitle_highlight_color=subtitle_style["subtitle_highlight_color"],
                subtitle_font_size=subtitle_style["subtitle_font_size"],
                subtitle_outline_size=subtitle_style["subtitle_outline_size"],
                subtitle_shadow_size=subtitle_style["subtitle_shadow_size"],
                subtitle_margin_v=subtitle_style["subtitle_margin_v"],
                subtitle_spacing=subtitle_style["subtitle_spacing"],
                subtitle_base_font=subtitle_style["subtitle_base_font"],
                subtitle_accent_font=subtitle_style["subtitle_accent_font"],
                subtitle_accent_last_word=subtitle_style["subtitle_accent_last_word"],
                progress_callback=progress_callback,
            )
            if result["payload"].get("subtitle_source") == "auto" and result["payload"].get("device") != st.session_state.get("device", "cuda"):
                st.warning(t("cuda_fallback_warning"))
            st.session_state.subtitle_burn_last_result = result
            progress_bar.progress(100)
            progress_status.success(t("progress_burn_done"))
        except Exception as exc:  # pragma: no cover - UI feedback
            if progress_status is not None:
                progress_status.error(str(exc))
            st.error(str(exc))

    subtitle_burn_result = st.session_state.get("subtitle_burn_last_result")
    if subtitle_burn_result:
        _render_subtitle_burn_result(subtitle_burn_result)


def _render_shorts() -> None:
    st.subheader(t("video_header"))
    if st.button(t("back_home")):
        _set_view("home")
        _rerun()

    uploaded_video = st.file_uploader(t("upload_video"), type=["mp4", "mov", "mkv"], help=t("upload_video_help"))
    st.write(t("video_intro"))

    shorts = []
    for index in range(2):
        with st.expander(f"{t('short_title')} {index + 1}", expanded=index == 0):
            title = st.text_input(f"{t('short_title')} {index + 1}", key=f"title_{index}")
            description = st.text_area(f"{t('short_description')} {index + 1}", key=f"description_{index}", height=100)
            target_text = st.text_area(
                f"{t('short_excerpt')} {index + 1}",
                key=f"target_{index}",
                height=180,
                placeholder=t("short_excerpt_placeholder"),
            )
            shorts.append(
                {
                    "title": title.strip(),
                    "description": description.strip(),
                    "target_text": target_text.strip(),
                }
            )

    col1, col2 = st.columns(2)
    with col1:
        start_padding_seconds = st.slider(
            t("start_padding"),
            0.0,
            2.0,
            DEFAULT_START_PADDING_SECONDS,
            step=0.05,
        )
        match_threshold = st.slider(t("match_threshold"), 50, 100, DEFAULT_MATCH_THRESHOLD)
    with col2:
        end_padding_seconds = st.slider(t("end_padding"), 0.0, 2.0, DEFAULT_END_PADDING_SECONDS, step=0.05)
    zoom_out = st.slider(t("zoom_out"), 0.0, 0.4, 0.0, step=0.05, help=t("zoom_out_help"))
    subtitle_style = _render_subtitle_style_controls()

    if st.button(t("generate_shorts"), type="primary", disabled=uploaded_video is None):
        st.session_state.shorts_last_results = []
        progress_bar = None
        progress_status = None
        try:
            progress_bar = st.progress(0)
            progress_status = st.empty()
            progress_status.info(t("progress_shorts_title"))
            progress_messages = {
                "loading_model": t("progress_shorts_loading_model"),
                "transcribing": t("progress_shorts_transcribing"),
                "transcription_cached": t("progress_shorts_transcription_cached"),
                "transcription_done": t("progress_shorts_transcribing"),
                "matching_excerpt": t("progress_shorts_matching_excerpt"),
                "building_subtitles": t("progress_shorts_building_subtitles"),
                "rendering_video": t("progress_shorts_rendering_video"),
                "building_thumbnail": t("progress_shorts_building_thumbnail"),
                "saving_metadata": t("progress_shorts_saving_metadata"),
            }
            progress_callback = _make_progress_updater(progress_bar, progress_status, progress_messages)
            input_path = _save_upload(uploaded_video, UPLOADS_DIR)
            generated_results = []
            for index, short in enumerate(shorts, start=1):
                if not short["title"] and not short["target_text"]:
                    continue
                if not short["target_text"]:
                    st.warning(f"{t('short_title')} {index} {t('short_skipped_empty')}")
                    continue
                result = generate_short_from_text(
                    input_path,
                    target_text=short["target_text"],
                    title=short["title"] or f"{t('short_title')} {index}",
                    description=short["description"],
                    output_dir=SHORTS_DIR,
                    model_name=st.session_state.get("model_name", DEFAULT_WHISPER_MODEL),
                    device=st.session_state.get("device", "cuda"),
                    start_padding_seconds=start_padding_seconds,
                    end_padding_seconds=end_padding_seconds,
                    start_threshold=match_threshold,
                    end_threshold=match_threshold,
                    subtitle_color=subtitle_style["subtitle_color"],
                    subtitle_highlight_color=subtitle_style["subtitle_highlight_color"],
                    subtitle_font_size=subtitle_style["subtitle_font_size"],
                    subtitle_outline_size=subtitle_style["subtitle_outline_size"],
                    subtitle_shadow_size=subtitle_style["subtitle_shadow_size"],
                    subtitle_margin_v=subtitle_style["subtitle_margin_v"],
                    subtitle_spacing=subtitle_style["subtitle_spacing"],
                    subtitle_base_font=subtitle_style["subtitle_base_font"],
                    subtitle_accent_font=subtitle_style["subtitle_accent_font"],
                    subtitle_accent_last_word=subtitle_style["subtitle_accent_last_word"],
                    zoom_out=zoom_out,
                    progress_callback=progress_callback,
                )
                if not result["payload"].get("cache_hit") and result["payload"].get("device") != st.session_state.get("device", "cuda"):
                    st.warning(t("cuda_fallback_warning"))
                st.success(f"{t('short_title')} {index}: {t('short_generated')}")
                generated_results.append(result)
            st.session_state.shorts_last_results = generated_results
            progress_bar.progress(100)
            progress_status.success(t("progress_shorts_done"))
        except Exception as exc:  # pragma: no cover - UI feedback
            if progress_status is not None:
                progress_status.error(str(exc))
            st.error(str(exc))

    shorts_results = st.session_state.get("shorts_last_results", [])
    if shorts_results:
        _render_shorts_results(shorts_results)


_inject_styles()

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "pt"
if "view" not in st.session_state:
    st.session_state.view = "home"
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_WHISPER_MODEL
if "device" not in st.session_state:
    st.session_state.device = "cuda"

_render_sidebar()

if st.session_state.view == "clean_audio":
    _render_clean_audio()
elif st.session_state.view == "transcription":
    _render_transcription()
elif st.session_state.view == "shorts":
    _render_shorts()
elif st.session_state.view == "subtitle_burn":
    _render_subtitle_burn()
else:
    _render_home()
