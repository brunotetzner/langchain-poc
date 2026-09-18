from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from flask import Blueprint, jsonify


licitacao_controller = Blueprint("licitacao", __name__, url_prefix="/licitacoes")


def load_service():
    service_path = (
        Path(__file__).parent.parent / "services" / "licitacao.service.py"
    )
    spec = spec_from_file_location("licitacao_service", service_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {service_path}")

    service = module_from_spec(spec)
    spec.loader.exec_module(service)
    return service


@licitacao_controller.get("")
def get_licitacoes():
    return jsonify(load_service().get_licitacoes())
