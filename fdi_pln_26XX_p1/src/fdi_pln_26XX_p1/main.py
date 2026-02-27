import click
import uvicorn


@click.command()
@click.option("--port", default=8001, help="Puerto donde escuchará el agente")
def cli(port):
    """Punto de entrada principal para arrancar el servidor del agente."""
    import logging

    logging.info(f"Arrancando agente en el puerto {port}...")

    # servidor FastAPI
    uvicorn.run("fdi_pln_26XX_p1.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    cli()
