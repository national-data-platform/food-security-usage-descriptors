import json

import requests

from src import Configuration


class FinishProcessNotificationUseCase:
    def __init__(self):
        self.config = Configuration().get()

    def execute(self, message) -> None:
        data = json.loads(message['data'])
        url = data['webhook_url']
        try:
            response = requests.post(
                url,
                data=json.dumps({
                    "taskId": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                    "results": {
                        "summary": {
                            "num_publications": 12504,
                            "num_authors": 34012,
                            "num_citations": 88023,
                            "num_journals": 432
                        },
                        "connection_details": {
                            "mongodb_connection_string": "mongodb://host:port/",
                            "mongodb_database": "dataset_a1b2c3d4"
                        },
                        "dashboard_url": "https://dashboard.yourdomain.com/dashboards/123?db_filter=dataset_a1b2c3d4"
                    }
                }),
                timeout=10
            )
            response.raise_for_status()
            if response.status_code > 299:
                return response.json()
            return None
        except requests.exceptions.HTTPError as http_err:
            raise Exception(f"Erro HTTP: {http_err}, Detalhes: {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception("Erro de conexão: Não foi possível conectar ao servidor")
        except requests.exceptions.Timeout:
            raise Exception("Erro de timeout: A requisição demorou demais")
        except ValueError:
            raise Exception("Erro: Resposta não é JSON válido")
