import click
import uvicorn
from loguru import logger


@click.command()
@click.option("--port", default=8001, help="Puerto donde escuchará el agente.")
def cli(port: int) -> None:
    """Arranca el servidor del agente de trueque."""
    logger.info("Iniciando agente en el puerto {}", port)
    uvicorn.run("fdi_pln_2614_p1.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    cli()
