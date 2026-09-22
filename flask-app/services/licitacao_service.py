from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TypedDict

from integrations.pncp import (
    ArquivoLicitacao,
    Licitacao,
    baixar_pdf,
    buscar_licitacoes,
    buscar_editais
)
from internal_services.rag_service import process_pdf


class ResultadoAnaliseLicitacao(TypedDict):
    licitacao: Licitacao
    arquivo: ArquivoLicitacao
    respostas: list[str]


def get_licitacoes() -> list[ResultadoAnaliseLicitacao]:
    """Busca três licitações e analisa seus editais sequencialmente."""
    resultados: list[ResultadoAnaliseLicitacao] = []

    licitacoes = buscar_licitacoes()
    print("+++++ LICITAÇÕES ENCONTRADAS +++++")
    print(licitacoes)
    for licitacao in licitacoes:
        editais = buscar_editais(licitacao)

        for arquivo in editais:
            pdf_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="pncp_",
                    suffix=".pdf",
                    delete=False,
                ) as temporario:
                    pdf_path = Path(temporario.name)

                print(f"Baixando PDF para {pdf_path, arquivo}...")
                baixar_pdf(arquivo, pdf_path)
                # respostas = analisar_pdf(str(pdf_path))
                respostas = ["skipped"]
                resultados.append(
                    {
                        "licitacao": licitacao,
                        "arquivo": arquivo,
                        "respostas": respostas,
                    }
                )
            finally:
                if pdf_path is not None:
                    pdf_path.unlink(missing_ok=True)

    return resultados


def analisar_pdf(pdf_path: str, perguntas: list[str] | None = None) -> list[str]:
    """
    Recebe o caminho de um PDF, delega ao serviço RAG e retorna as respostas.

    Args:
        pdf_path: Caminho absoluto ou relativo para o arquivo PDF.
        perguntas: Lista opcional de perguntas. Se None, usa as padrão.

    Returns:
        list[str]: Respostas geradas pelo agente RAG.
    """
    return process_pdf(pdf_path, perguntas)
