from __future__ import annotations

import asyncio

from code_agent.egress import EgressProxy, parse_allowlist


def test_parse_allowlist_defaults_to_443():
    assert parse_allowlist(["API.example.com", "localhost:8443"]) == {
        ("api.example.com", 443),
        ("localhost", 8443),
    }


async def _exchange(proxy_port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(request)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(4096), 5)
    writer.close()
    return data


def test_tunnels_allowed_and_blocks_everything_else():
    async def scenario() -> None:
        async def echo(reader, writer):
            writer.write(b"echo:" + await reader.read(100))
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        up_port = upstream.sockets[0].getsockname()[1]
        proxy = EgressProxy(parse_allowlist([f"127.0.0.1:{up_port}"]))
        server = await proxy.serve("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        # Allowed: tunnel established, bytes flow both ways.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT 127.0.0.1:{up_port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        await writer.drain()
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
        writer.write(b"hello")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(100), 5) == b"echo:hello"
        writer.close()

        # Other host, other port on the allowed host, plain HTTP, garbage: all refused.
        denied = await _exchange(port, b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
        assert denied.startswith(b"HTTP/1.1 403")
        other_port = await _exchange(port, b"CONNECT 127.0.0.1:22 HTTP/1.1\r\n\r\n")
        assert other_port.startswith(b"HTTP/1.1 403")
        plain = await _exchange(port, b"GET http://evil.example/ HTTP/1.1\r\n\r\n")
        assert plain.startswith(b"HTTP/1.1 405")
        assert proxy.denied == ["evil.example:443", "127.0.0.1:22"]
        assert proxy.tunnelled == [f"127.0.0.1:{up_port}"]

        server.close()
        upstream.close()

    asyncio.run(scenario())
