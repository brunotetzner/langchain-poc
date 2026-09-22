from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from integrations.pncp.config import PNCP_API_URL, PNCP_COOKIE, REQUEST_TIMEOUT_SECONDS


PNCP_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Cookie": PNCP_COOKIE,
}


class Licitacao(TypedDict, total=False):
    id: str
    title: str
    description: str
    item_url: str
    document_type: str
    ano: str
    numero_sequencial: str
    numero_controle_pncp: str
    orgao_cnpj: str
    orgao_nome: str


class ArquivoLicitacao(TypedDict, total=False):
    uri: str
    url: str
    tipoDocumentoId: int
    statusAtivo: bool
    cnpj: str
    anoCompra: int
    sequencialCompra: int
    sequencialDocumento: int
    titulo: str
    tipoDocumentoNome: str
    tipoDocumentoDescricao: str


def _create_http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            nome: valor
            for nome, valor in PNCP_HEADERS.items()
            if valor
        }
    )
    session.mount("https://", adapter)
    return session


def _get_json(url: str, *, params: dict[str, Any]) -> list[dict[str, Any]]:
    response: requests.Response | None = None
    try:
        with _create_http_session() as session:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        print(
            f"Erro na requisição HTTP para {url} "
            f"(status={getattr(response, 'status_code', 'desconhecido')}): {error}"
        )
        raise
    except ValueError as error:
        print(
            f"Resposta não-JSON do PNCP para {url} "
            f"(status={getattr(response, 'status_code', 'desconhecido')}): "
            f"{getattr(response, 'text', '')[:1000]}"
        )
        raise ValueError(f"Resposta inválida da API do PNCP: {url}") from error

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    else:
        print(
            f"Resposta inesperada do PNCP para {url} "
            f"(status={response.status_code}): {response.text[:1000]}"
        )
        raise ValueError(f"Resposta inesperada da API do PNCP: {url}")

    if not all(isinstance(item, dict) for item in items):
        print(
            f"Itens inválidos na resposta do PNCP para {url}: "
            f"{getattr(response, 'text', '')[:1000]}"
        )
        raise ValueError(f"Itens inválidos na resposta da API do PNCP: {url}")

    return cast(list[dict[str, Any]], items)


def buscar_licitacoes() -> list[Licitacao]:
    payload = _get_json(
        f"{PNCP_API_URL}/search/",
        params={
            "tipos_documento": "edital",
            "ordenacao": "-data",
            "pagina": 1,
            "tam_pagina": 3,
            "status": "recebendo_proposta",
        },
    )
    return cast(list[Licitacao], payload[:3])


def buscar_licitacao_arquivos(licitacao: Licitacao) -> list[ArquivoLicitacao]:
    cnpj = licitacao.get("orgao_cnpj")
    ano = licitacao.get("ano")
    sequencial = licitacao.get("numero_sequencial")
    if not cnpj or not ano or not sequencial:
        raise ValueError("Licitação sem órgão, ano ou número sequencial.")

    payload = _get_json(
        f"{PNCP_API_URL}/pncp/v1/orgaos/{cnpj}/compras/{ano}/{sequencial}/arquivos",
        params={"pagina": 1, "tamanhoPagina": 5},
    )
    return cast(list[ArquivoLicitacao], payload)

def buscar_editais(licitacao: Licitacao) -> list[ArquivoLicitacao]:
    arquivos = buscar_licitacao_arquivos(licitacao)

    termos_referencia = [
        arquivo
        for arquivo in arquivos
        if arquivo.get("tipoDocumentoNome", "").casefold()
        == "termo de referência"
    ]

    if termos_referencia:
        return termos_referencia

    return [
        arquivo
        for arquivo in arquivos
        if arquivo.get("tipoDocumentoNome", "").casefold() == "edital"
    ]

def baixar_pdf(arquivo: ArquivoLicitacao, destino: Path) -> None:
    url = arquivo.get("url") or arquivo.get("uri")
    if not url:
        raise ValueError("Arquivo do PNCP sem URL para download.")

    try:
        with _create_http_session() as session:
            with session.get(
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            ) as response:
                response.raise_for_status()
                with destino.open("wb") as pdf:
                    for bloco in response.iter_content(chunk_size=1024 * 1024):
                        if bloco:
                            pdf.write(bloco)
    except requests.RequestException as error:
        print(f"Erro ao baixar o PDF {url}: {error}")
        raise