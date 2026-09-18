from controllers.licitacao_controller import licitacao_controller


def register_routes(app):
    app.register_blueprint(licitacao_controller)