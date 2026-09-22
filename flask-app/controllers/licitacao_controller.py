from pathlib import Path

from flask import Blueprint, jsonify, request

from services.licitacao_service import analisar_pdf, get_licitacoes


licitacao_controller = Blueprint("licitacao", __name__, url_prefix="/licitacoes")


@licitacao_controller.post("")
def listar_licitacoes():
    return jsonify(get_licitacoes())


@licitacao_controller.post("/analisar")
def analisar_pdf_route():
    """
    Recebe um PDF via upload, delega ao serviço RAG e retorna as respostas.
    Espera um multipart/form-data com o campo 'pdf' contendo o arquivo
    e um campo opcional 'perguntas' (JSON array de strings).
    """
    if "pdf" not in request.files:
        return jsonify({"erro": "Campo 'pdf' é obrigatório."}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    temp_path = Path("/tmp") / pdf_file.filename
    pdf_file.save(str(temp_path))

    perguntas_raw = request.form.get("perguntas")
    perguntas: list[str] | None = None
    if perguntas_raw:
        import json
        perguntas = json.loads(perguntas_raw)

    try:
        respostas = analisar_pdf(str(temp_path), perguntas)
        return jsonify({"respostas": respostas})
    finally:
        if temp_path.exists():
            temp_path.unlink()

