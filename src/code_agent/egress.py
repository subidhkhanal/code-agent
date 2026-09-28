"""Allowlisting egress proxy for headless containers (ADR 0009).

The agent container sits on an internal Docker network with no route to the internet. Its only
way out is this proxy, which runs in a second container attached to both networks and tunnels
HTTPS `CONNECT` requests to allowlisted `host:port` pairs only (by default, the LLM provider's API
host on 443). Everything else gets `403`. TLS stays end-to-end: the proxy sees host names, never
request contents, and holds no credentials.

    python -m code_agent.egress --allow generativelanguage.googleapis.com:443 --port 3128

httpx honours `HTTPS_PROXY`, so the agent needs no code changes to use it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

log = logging.getLogger("code_agent.egress")

MAX_HEADER_BYTES = 8192
HEADER_TIMEOUT_S = 10.0


def parse_allowlist(entries: list[str]) -> frozenset[tuple[str, int]]:
    allowed = set()
    for entry in entries:
        host, _, port = entry.strip().lower().rpartition(":")
        if not host:  # no port given
            host, port = entry.strip().lower(), "443"
        allowed.add((host, int(port)))
    return frozenset(allowed)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


class EgressProxy:
    def __init__(self, allowed: frozenset[tuple[str, int]]) -> None:
        self.allowed = allowed
        self.denied: list[str] = []  # for tests and the audit trail
        self.tunnelled: list[str] = []

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEADER_TIMEOUT_S)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
            writer.close()
            return
        request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = request_line.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            await self._reply(writer, 405, "only CONNECT (HTTPS) is allowed")
            return
        host, _, port_text = parts[1].lower().rpartition(":")
        try:
            target = (host.strip("[]"), int(port_text))
        except ValueError:
            await self._reply(writer, 400, "bad CONNECT target")
            return
        if target not in self.allowed:
            self.denied.append(f"{target[0]}:{target[1]}")
            log.warning("denied CONNECT %s:%s", *target)
            await self._reply(writer, 403, "destination not on the egress allowlist")
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(*target), HEADER_TIMEOUT_S
            )
        except (OSError, TimeoutError):
            await self._reply(writer, 502, "upstream unreachable")
            return
        self.tunnelled.append(f"{target[0]}:{target[1]}")
        log.info("tunnel %s:%s", *target)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
        body = reason.encode()
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n".encode()
            + body
        )
        with contextlib.suppress(ConnectionError):
            await writer.drain()
        writer.close()

    async def serve(self, host: str, port: int) -> asyncio.Server:
        return await asyncio.start_server(self.handle, host, port, limit=MAX_HEADER_BYTES)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Allowlisting HTTPS egress proxy")
    ap.add_argument("--allow", action="append", required=True, help="host[:port], repeatable")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=3128)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(message)s")
    proxy = EgressProxy(parse_allowlist(args.allow))

    async def run() -> None:
        server = await proxy.serve(args.host, args.port)
        log.info("egress proxy on %s:%s allowing %s", args.host, args.port, sorted(proxy.allowed))
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
