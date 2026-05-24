from loguru import logger
import httpx


class ButlerClient:
    """Thin async wrapper around the Butler REST API."""

    def __init__(self, base_url: str, alias: str, http: httpx.AsyncClient) -> None:
        self.base_url = base_url
        self.alias = alias
        self._http = http

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _params(self) -> dict[str, str]:
        return {"agente": self.alias}

    async def register(self) -> None:
        await self._http.post(self._url(f"/alias/{self.alias}"), params=self._params())
        logger.info("Registered alias '{}'", self.alias)

    async def get_info(self) -> dict:
        r = await self._http.get(self._url("/info"), params=self._params())
        r.raise_for_status()
        return r.json()

    async def get_agents(self) -> list[str]:
        r = await self._http.get(self._url("/gente"), params=self._params())
        if r.status_code != 200:
            return []
        raw: list = r.json()
        names = [g.get("alias") if isinstance(g, dict) else g for g in raw]
        return [n for n in names if n and n not in (self.alias, "System", "Sistema")]

    async def send_message(self, dest: str, subject: str, body: str) -> None:
        await self._http.post(
            self._url("/carta"),
            params=self._params(),
            json={"remi": self.alias, "dest": dest, "asunto": subject, "cuerpo": body},
        )
        logger.debug("→ {} | {:.80}", dest, body)

    async def delete_message(self, uid: str) -> None:
        await self._http.delete(self._url(f"/mail/{uid}"), params=self._params())

    async def get_agent_info(self, alias: str) -> dict:
        r = await self._http.get(self._url("/info"), params={"agente": alias})
        r.raise_for_status()
        return r.json()

    async def send_package(self, dest: str, items: dict[str, int]) -> None:
        await self._http.post(
            self._url(f"/paquete/{dest}"),
            params=self._params(),
            json=items,
        )
        logger.info("Package sent to {}: {}", dest, items)
